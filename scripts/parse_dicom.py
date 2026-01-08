#!/usr/bin/env python3
"""
DICOM Directory Parser
======================
Parses DICOM directories, extracts images as PNGs in a hierarchical structure,
and generates comprehensive JSON metadata.

Usage:
    python parse_dicom.py --input /path/to/dicom/dir --output /path/to/output/dir

Structure Output:
    output_dir/
    ├── patient_name/
    │   ├── study_date_description/
    │   │   ├── series_number_description/
    │   │   │   ├── slice_0001.png
    │   │   │   ├── slice_0002.png
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
    └── metadata.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    print("Error: pydicom is required. Install with: pip install pydicom")
    sys.exit(1)

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("Error: numpy and Pillow are required. Install with: pip install numpy Pillow")
    sys.exit(1)


def sanitize_filename(name: str, max_length: int = 50) -> str:
    """Sanitize a string to be used as a filename."""
    if not name:
        return "unknown"
    # Replace problematic characters
    name = re.sub(r'[<>:"/\\|?*^]', '_', str(name))
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    # Truncate if too long
    if len(name) > max_length:
        name = name[:max_length]
    return name or "unknown"


def format_date(date_str: Optional[str]) -> str:
    """Convert DICOM date format (YYYYMMDD) to readable format."""
    if not date_str:
        return "unknown_date"
    try:
        dt = datetime.strptime(str(date_str), "%Y%m%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return str(date_str)


def format_time(time_str: Optional[str]) -> str:
    """Convert DICOM time format to readable format."""
    if not time_str:
        return ""
    try:
        time_str = str(time_str).split('.')[0]  # Remove fractional seconds
        if len(time_str) >= 6:
            dt = datetime.strptime(time_str[:6], "%H%M%S")
            return dt.strftime("%H:%M:%S")
        return time_str
    except ValueError:
        return str(time_str)


def get_dicom_value(ds: pydicom.Dataset, tag: str, default: Any = None) -> Any:
    """Safely get a DICOM tag value."""
    try:
        if hasattr(ds, tag):
            val = getattr(ds, tag)
            if val is not None:
                # Handle PersonName objects
                if hasattr(val, 'family_name') or hasattr(val, 'given_name'):
                    return str(val)
                return val
        return default
    except Exception:
        return default


def normalize_pixel_array(pixel_array: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """Normalize pixel array to 8-bit for PNG output with proper windowing."""
    # Apply rescale slope and intercept if present
    slope = float(get_dicom_value(ds, 'RescaleSlope', 1))
    intercept = float(get_dicom_value(ds, 'RescaleIntercept', 0))
    
    pixels = pixel_array.astype(np.float64) * slope + intercept
    
    # Apply window center/width if available
    window_center = get_dicom_value(ds, 'WindowCenter', None)
    window_width = get_dicom_value(ds, 'WindowWidth', None)
    
    if window_center is not None and window_width is not None:
        # Handle multi-value window settings
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = float(window_center[0])
        else:
            window_center = float(window_center)
            
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = float(window_width[0])
        else:
            window_width = float(window_width)
        
        # Apply windowing
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        pixels = np.clip(pixels, lower, upper)
        pixels = ((pixels - lower) / (upper - lower) * 255).astype(np.uint8)
    else:
        # Simple min-max normalization
        min_val = pixels.min()
        max_val = pixels.max()
        if max_val > min_val:
            pixels = ((pixels - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            pixels = np.zeros_like(pixels, dtype=np.uint8)
    
    # Handle photometric interpretation
    photometric = get_dicom_value(ds, 'PhotometricInterpretation', 'MONOCHROME2')
    if photometric == 'MONOCHROME1':
        pixels = 255 - pixels
    
    return pixels


def determine_imaging_plane(orientation: List[float], modality: str = 'MR') -> str:
    """
    Determine the imaging plane (Axial, Sagittal, Coronal) from ImageOrientationPatient.
    
    ImageOrientationPatient contains 6 values: [row_x, row_y, row_z, col_x, col_y, col_z]
    - Row vector (first 3): direction cosines of the first row of the image
    - Column vector (last 3): direction cosines of the first column of the image
    
    The cross product of row and column vectors gives the normal to the imaging plane.
    In the DICOM Patient Coordinate System:
    - X: increases to patient's left (Right-to-Left)
    - Y: increases to patient's posterior (Anterior-to-Posterior)
    - Z: increases toward patient's head (Inferior-to-Superior)
    
    Imaging planes based on normal vector direction:
    - Axial (transverse): normal along Z-axis (slices top-to-bottom)
    - Sagittal: normal along X-axis (slices left-to-right, side view)
    - Coronal: normal along Y-axis (slices front-to-back, front view)
    
    Args:
        orientation: List of 6 floats from ImageOrientationPatient
        modality: DICOM modality (MR, DX, CT, etc.)
    
    Returns:
        String indicating the imaging plane: 'Axial', 'Sagittal', 'Coronal', 'Oblique', 
        'Projection' (for DX/CR), or 'Unknown'
    """
    # For projection radiography (DX, CR), there's no 3D orientation
    if modality in ('DX', 'CR', 'DR'):
        # DX images are typically projection images, not volumetric slices
        # We can try to infer view from ViewPosition or other tags
        return 'Projection'
    
    if not orientation or len(orientation) != 6:
        return 'Unknown'
    
    try:
        # Row and column direction cosines
        row = np.array(orientation[:3], dtype=np.float64)
        col = np.array(orientation[3:6], dtype=np.float64)
        
        # Normal vector (perpendicular to the image plane)
        normal = np.cross(row, col)
        
        # Normalize the normal vector
        norm_magnitude = np.linalg.norm(normal)
        if norm_magnitude < 1e-6:
            return 'Unknown'
        normal = normal / norm_magnitude
        
        # Get absolute values to find dominant axis
        abs_normal = np.abs(normal)
        dominant_axis = np.argmax(abs_normal)
        dominant_value = abs_normal[dominant_axis]
        
        # Threshold for determining if the plane is oblique
        # If the dominant component is less than 0.8, it's an oblique plane
        OBLIQUE_THRESHOLD = 0.8
        
        if dominant_value < OBLIQUE_THRESHOLD:
            return 'Oblique'
        
        # Determine plane based on dominant axis of normal vector
        if dominant_axis == 0:  # Normal along X-axis -> Sagittal
            return 'Sagittal'
        elif dominant_axis == 1:  # Normal along Y-axis -> Coronal
            return 'Coronal'
        else:  # dominant_axis == 2, Normal along Z-axis -> Axial
            return 'Axial'
            
    except Exception:
        return 'Unknown'


def extract_patient_info(ds: pydicom.Dataset) -> Dict[str, Any]:
    """Extract patient-level information from DICOM dataset."""
    patient_name = get_dicom_value(ds, 'PatientName', 'Unknown')
    if patient_name:
        patient_name = str(patient_name).replace('^', ' ').strip()
    
    return {
        'patient_name': patient_name,
        'patient_id': get_dicom_value(ds, 'PatientID', ''),
        'patient_sex': get_dicom_value(ds, 'PatientSex', ''),
        'patient_birth_date': format_date(get_dicom_value(ds, 'PatientBirthDate', '')),
        'patient_age': get_dicom_value(ds, 'PatientAge', ''),
        'patient_weight': get_dicom_value(ds, 'PatientWeight', ''),
    }


def extract_station_info(ds: pydicom.Dataset) -> Dict[str, Any]:
    """Extract station/equipment information from DICOM dataset."""
    return {
        'manufacturer': get_dicom_value(ds, 'Manufacturer', ''),
        'manufacturer_model_name': get_dicom_value(ds, 'ManufacturerModelName', ''),
        'station_name': get_dicom_value(ds, 'StationName', ''),
        'software_versions': get_dicom_value(ds, 'SoftwareVersions', ''),
        'device_serial_number': get_dicom_value(ds, 'DeviceSerialNumber', ''),
        'magnetic_field_strength': get_dicom_value(ds, 'MagneticFieldStrength', ''),
    }


def extract_study_info(ds: pydicom.Dataset) -> Dict[str, Any]:
    """Extract study-level information from DICOM dataset."""
    return {
        'study_instance_uid': get_dicom_value(ds, 'StudyInstanceUID', ''),
        'study_date': format_date(get_dicom_value(ds, 'StudyDate', '')),
        'study_time': format_time(get_dicom_value(ds, 'StudyTime', '')),
        'study_id': get_dicom_value(ds, 'StudyID', ''),
        'accession_number': get_dicom_value(ds, 'AccessionNumber', ''),
        'study_description': get_dicom_value(ds, 'StudyDescription', ''),
        'referring_physician': str(get_dicom_value(ds, 'ReferringPhysicianName', '')).replace('^', ' ').strip(),
        'institution_name': get_dicom_value(ds, 'InstitutionName', ''),
        'institution_department': get_dicom_value(ds, 'InstitutionalDepartmentName', ''),
    }


def extract_series_info(ds: pydicom.Dataset) -> Dict[str, Any]:
    """Extract series-level information from DICOM dataset."""
    return {
        'series_instance_uid': get_dicom_value(ds, 'SeriesInstanceUID', ''),
        'series_date': format_date(get_dicom_value(ds, 'SeriesDate', '')),
        'series_time': format_time(get_dicom_value(ds, 'SeriesTime', '')),
        'series_number': get_dicom_value(ds, 'SeriesNumber', 0),
        'series_description': get_dicom_value(ds, 'SeriesDescription', ''),
        'modality': get_dicom_value(ds, 'Modality', ''),
        'body_part_examined': get_dicom_value(ds, 'BodyPartExamined', ''),
        'laterality': get_dicom_value(ds, 'Laterality', ''),
        'protocol_name': get_dicom_value(ds, 'ProtocolName', ''),
    }


def extract_acquisition_info(ds: pydicom.Dataset) -> Dict[str, Any]:
    """Extract MRI acquisition parameters."""
    return {
        'scanning_sequence': get_dicom_value(ds, 'ScanningSequence', ''),
        'sequence_variant': get_dicom_value(ds, 'SequenceVariant', ''),
        'scan_options': get_dicom_value(ds, 'ScanOptions', ''),
        'mr_acquisition_type': get_dicom_value(ds, 'MRAcquisitionType', ''),
        'repetition_time': get_dicom_value(ds, 'RepetitionTime', ''),
        'echo_time': get_dicom_value(ds, 'EchoTime', ''),
        'echo_train_length': get_dicom_value(ds, 'EchoTrainLength', ''),
        'flip_angle': get_dicom_value(ds, 'FlipAngle', ''),
        'number_of_averages': get_dicom_value(ds, 'NumberOfAverages', ''),
        'imaging_frequency': get_dicom_value(ds, 'ImagingFrequency', ''),
        'imaged_nucleus': get_dicom_value(ds, 'ImagedNucleus', ''),
        'receive_coil_name': get_dicom_value(ds, 'ReceiveCoilName', ''),
        'patient_position': get_dicom_value(ds, 'PatientPosition', ''),
    }


def extract_image_info(ds: pydicom.Dataset) -> Dict[str, Any]:
    """Extract image-level information from DICOM dataset."""
    # Handle image orientation - can be a list
    orientation = get_dicom_value(ds, 'ImageOrientationPatient', [])
    if hasattr(orientation, '__iter__') and not isinstance(orientation, str):
        orientation = [float(x) for x in orientation]
    
    # Handle image position - can be a list
    position = get_dicom_value(ds, 'ImagePositionPatient', [])
    if hasattr(position, '__iter__') and not isinstance(position, str):
        position = [float(x) for x in position]
    
    # Handle pixel spacing
    pixel_spacing = get_dicom_value(ds, 'PixelSpacing', [])
    if hasattr(pixel_spacing, '__iter__') and not isinstance(pixel_spacing, str):
        pixel_spacing = [float(x) for x in pixel_spacing]
    
    return {
        'sop_instance_uid': get_dicom_value(ds, 'SOPInstanceUID', ''),
        'sop_class_uid': get_dicom_value(ds, 'SOPClassUID', ''),
        'instance_number': get_dicom_value(ds, 'InstanceNumber', 0),
        'image_type': get_dicom_value(ds, 'ImageType', ''),
        'photometric_interpretation': get_dicom_value(ds, 'PhotometricInterpretation', ''),
        'samples_per_pixel': get_dicom_value(ds, 'SamplesPerPixel', 1),
        'rows': get_dicom_value(ds, 'Rows', 0),
        'columns': get_dicom_value(ds, 'Columns', 0),
        'bits_allocated': get_dicom_value(ds, 'BitsAllocated', 0),
        'bits_stored': get_dicom_value(ds, 'BitsStored', 0),
        'pixel_representation': get_dicom_value(ds, 'PixelRepresentation', 0),
        'pixel_spacing': pixel_spacing,
        'slice_thickness': get_dicom_value(ds, 'SliceThickness', ''),
        'slice_location': get_dicom_value(ds, 'SliceLocation', ''),
        'spacing_between_slices': get_dicom_value(ds, 'SpacingBetweenSlices', ''),
        'image_position_patient': position,
        'image_orientation_patient': orientation,
        'window_center': get_dicom_value(ds, 'WindowCenter', ''),
        'window_width': get_dicom_value(ds, 'WindowWidth', ''),
        'acquisition_number': get_dicom_value(ds, 'AcquisitionNumber', ''),
    }


def is_image_dicom(ds: pydicom.Dataset) -> bool:
    """Check if DICOM file contains pixel data (is an image)."""
    return hasattr(ds, 'PixelData') and hasattr(ds, 'Rows') and hasattr(ds, 'Columns')


def find_dicom_files(input_dir: Path) -> List[Path]:
    """Recursively find all DICOM files in directory."""
    dicom_files = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = Path(root) / file
            # Skip non-DICOM files based on common patterns
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.txt', '.xml', '.json', '.exe')):
                continue
            dicom_files.append(file_path)
    
    return dicom_files


def create_series_dir_name(series_info: Dict[str, Any]) -> str:
    """Create a descriptive directory name for a series."""
    series_num = series_info.get('series_number', 0)
    description = series_info.get('series_description', '')
    modality = series_info.get('modality', '')
    
    parts = [f"ser_{series_num:04d}"]
    if description:
        parts.append(sanitize_filename(description, 30))
    if modality:
        parts.append(modality)
    
    return '_'.join(parts)


def create_study_dir_name(study_info: Dict[str, Any]) -> str:
    """Create a descriptive directory name for a study."""
    date = study_info.get('study_date', 'unknown_date')
    description = study_info.get('study_description', '')
    
    parts = [date]
    if description:
        parts.append(sanitize_filename(description, 40))
    
    return '_'.join(parts)


def create_patient_dir_name(patient_info: Dict[str, Any]) -> str:
    """Create a descriptive directory name for a patient."""
    name = patient_info.get('patient_name', 'Unknown')
    patient_id = patient_info.get('patient_id', '')
    
    dir_name = sanitize_filename(name, 40)
    if patient_id:
        dir_name = f"{dir_name}_{sanitize_filename(patient_id, 15)}"
    
    return dir_name


def parse_dicom_directory(input_dir: Path, output_dir: Path, verbose: bool = True) -> Dict[str, Any]:
    """
    Parse DICOM directory and create organized PNG output.
    
    Returns comprehensive metadata dictionary.
    """
    if verbose:
        print(f"Scanning DICOM directory: {input_dir}")
    
    # Find all DICOM files
    dicom_files = find_dicom_files(input_dir)
    if verbose:
        print(f"Found {len(dicom_files)} potential DICOM files")
    
    # Organize files by Patient > Study > Series
    hierarchy: Dict[str, Dict] = {}
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for file_path in dicom_files:
        try:
            ds = pydicom.dcmread(file_path, force=True)
            
            # Skip non-image DICOM files
            if not is_image_dicom(ds):
                skipped_count += 1
                continue
            
            # Extract metadata
            patient_info = extract_patient_info(ds)
            study_info = extract_study_info(ds)
            series_info = extract_series_info(ds)
            image_info = extract_image_info(ds)
            acquisition_info = extract_acquisition_info(ds)
            station_info = extract_station_info(ds)
            
            # Create hierarchy keys
            patient_key = patient_info['patient_id'] or patient_info['patient_name']
            study_key = study_info['study_instance_uid']
            series_key = series_info['series_instance_uid']
            
            # Initialize hierarchy levels
            if patient_key not in hierarchy:
                hierarchy[patient_key] = {
                    'info': patient_info,
                    'station': station_info,
                    'studies': {}
                }
            
            if study_key not in hierarchy[patient_key]['studies']:
                hierarchy[patient_key]['studies'][study_key] = {
                    'info': study_info,
                    'series': {}
                }
            
            if series_key not in hierarchy[patient_key]['studies'][study_key]['series']:
                hierarchy[patient_key]['studies'][study_key]['series'][series_key] = {
                    'info': series_info,
                    'acquisition': acquisition_info,
                    'images': [],
                    'first_orientation': image_info.get('image_orientation_patient', [])
                }
            
            # Add image to series
            hierarchy[patient_key]['studies'][study_key]['series'][series_key]['images'].append({
                'file_path': str(file_path),
                'info': image_info
            })
            
            processed_count += 1
            
        except InvalidDicomError:
            skipped_count += 1
        except Exception as e:
            error_count += 1
            if verbose:
                print(f"  Error processing {file_path}: {e}")
    
    if verbose:
        print(f"Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")
    
    # Create output structure and save PNGs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'parse_date': datetime.now().isoformat(),
        'source_directory': str(input_dir),
        'output_directory': str(output_dir),
        'summary': {
            'total_patients': len(hierarchy),
            'total_studies': 0,
            'total_series': 0,
            'total_images': 0
        },
        'patients': []
    }
    
    for patient_key, patient_data in hierarchy.items():
        patient_info = patient_data['info']
        station_info = patient_data['station']
        patient_dir_name = create_patient_dir_name(patient_info)
        patient_dir = output_dir / patient_dir_name
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        patient_metadata = {
            **patient_info,
            'station': station_info,
            'output_directory': patient_dir_name,
            'studies': []
        }
        
        for study_key, study_data in patient_data['studies'].items():
            study_info = study_data['info']
            study_dir_name = create_study_dir_name(study_info)
            study_dir = patient_dir / study_dir_name
            study_dir.mkdir(parents=True, exist_ok=True)
            
            metadata['summary']['total_studies'] += 1
            
            study_metadata = {
                **study_info,
                'output_directory': study_dir_name,
                'series': []
            }
            
            for series_key, series_data in study_data['series'].items():
                series_info = series_data['info']
                acquisition_info = series_data['acquisition']
                series_dir_name = create_series_dir_name(series_info)
                series_dir = study_dir / series_dir_name
                series_dir.mkdir(parents=True, exist_ok=True)
                
                metadata['summary']['total_series'] += 1
                
                # Sort images by instance number
                images = sorted(series_data['images'], 
                              key=lambda x: x['info'].get('instance_number', 0))
                
                # Determine imaging plane/view from first image orientation
                first_orientation = series_data.get('first_orientation', [])
                modality = series_info.get('modality', '')
                imaging_view = determine_imaging_plane(first_orientation, modality)
                
                series_metadata = {
                    **series_info,
                    'view': imaging_view,
                    'acquisition_parameters': acquisition_info,
                    'output_directory': series_dir_name,
                    'num_slices': len(images),
                    'slices': []
                }
                
                # Process each image
                for idx, image_data in enumerate(images):
                    try:
                        ds = pydicom.dcmread(image_data['file_path'], force=True)
                        pixel_array = ds.pixel_array
                        
                        # Normalize to 8-bit
                        normalized = normalize_pixel_array(pixel_array, ds)
                        
                        # Create PNG filename
                        instance_num = image_data['info'].get('instance_number', idx + 1)
                        png_filename = f"slice_{instance_num:04d}.png"
                        png_path = series_dir / png_filename
                        
                        # Save as PNG
                        img = Image.fromarray(normalized)
                        img.save(png_path)
                        
                        metadata['summary']['total_images'] += 1
                        
                        # Add slice metadata
                        slice_metadata = {
                            'filename': png_filename,
                            'instance_number': instance_num,
                            'slice_location': image_data['info'].get('slice_location', ''),
                            'image_position': image_data['info'].get('image_position_patient', []),
                            'pixel_spacing': image_data['info'].get('pixel_spacing', []),
                            'rows': image_data['info'].get('rows', 0),
                            'columns': image_data['info'].get('columns', 0),
                            'original_file': str(Path(image_data['file_path']).relative_to(input_dir)),
                        }
                        series_metadata['slices'].append(slice_metadata)
                        
                        if verbose and (idx + 1) % 10 == 0:
                            print(f"    Processed {idx + 1}/{len(images)} slices in {series_dir_name}")
                            
                    except Exception as e:
                        if verbose:
                            print(f"    Error saving image {image_data['file_path']}: {e}")
                
                # Calculate series-level image geometry
                if series_metadata['slices']:
                    first_slice = series_metadata['slices'][0]
                    series_metadata['image_dimensions'] = {
                        'rows': first_slice['rows'],
                        'columns': first_slice['columns'],
                        'pixel_spacing': first_slice['pixel_spacing'],
                        'slice_thickness': series_info.get('slice_thickness', series_data['acquisition'].get('slice_thickness', '')),
                    }
                
                study_metadata['series'].append(series_metadata)
                
                if verbose:
                    print(f"  Completed series: {series_dir_name} ({len(images)} slices)")
            
            patient_metadata['studies'].append(study_metadata)
            
            if verbose:
                print(f"Completed study: {study_dir_name}")
        
        metadata['patients'].append(patient_metadata)
        
        if verbose:
            print(f"Completed patient: {patient_dir_name}")
    
    # Save metadata JSON
    json_path = output_dir / 'metadata.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    
    if verbose:
        print(f"\nMetadata saved to: {json_path}")
        print(f"\nSummary:")
        print(f"  Patients: {metadata['summary']['total_patients']}")
        print(f"  Studies: {metadata['summary']['total_studies']}")
        print(f"  Series: {metadata['summary']['total_series']}")
        print(f"  Images: {metadata['summary']['total_images']}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Parse DICOM directory and extract images as PNGs with metadata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python parse_dicom.py --input ./DICOM --output ./parsed_mri
    python parse_dicom.py -i /path/to/dicom -o /path/to/output -q
    
Output Structure:
    output_dir/
    ├── Patient_Name_ID/
    │   ├── 2023-05-10_MRI_Knee/
    │   │   ├── ser_0301_PD_Axial_MR/
    │   │   │   ├── slice_0001.png
    │   │   │   └── ...
    │   │   └── ser_0401_T2_SPAIR_MR/
    │   │       └── ...
    │   └── ...
    └── metadata.json
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input DICOM directory path'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output directory path for PNGs and metadata'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    verbose = not args.quiet
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    try:
        metadata = parse_dicom_directory(input_dir, output_dir, verbose=verbose)
        
        if verbose:
            print("\nParsing completed successfully!")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

