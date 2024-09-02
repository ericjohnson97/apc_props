import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
from scipy.interpolate import interp1d
from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

def parse_data(filename: str) -> pd.DataFrame:
    """
    Parses the given file to extract and structure the data into a pandas DataFrame.
    
    Args:
        filename (str): The path to the file containing the data.
    
    Returns:
        pd.DataFrame: A DataFrame containing the parsed data.
    """
    data = []
    current_rpm = None
    headers = []
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            
            if line.startswith('PROP RPM'):
                current_rpm = int(re.findall(r'\d+', line)[0])
            elif re.match(r'V\s+', line):
                headers = re.split(r'\s{2,}', line)
                headers.insert(0, "PROP RPM")  # Add PROP RPM as the first column
            elif re.match(r'\(\w+\)', line):
                continue
            elif re.match(r'^\d', line):
                values = re.split(r'\s{2,}', line)
                try:
                    values = [float(v) for v in values]
                    values.insert(0, current_rpm)  # Insert the PROP RPM value at the beginning
                    data.append(values)
                except ValueError:
                    print(f"Skipping line: {line}")
                    continue
    
    df = pd.DataFrame(data, columns=headers)
    
    return df

def interpolate_and_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates and averages Ct and Cp values over a common range of J (Advance Ratio).
    
    Args:
        df (pd.DataFrame): The DataFrame containing the parsed data.
    
    Returns:
        pd.DataFrame: A DataFrame containing the averaged Ct and Cp values.
    """
    min_j = df['J'].min()
    max_j = df['J'].max()
    
    j_common = np.linspace(min_j, max_j, 20)
    
    averaged_data = {'J': j_common, 'Ct': [], 'Cp': []}
    
    rpm_values = df['PROP RPM'].unique()
    
    for j in j_common:
        ct_values = []
        cp_values = []
        
        for rpm in rpm_values:
            subset = df[df['PROP RPM'] == rpm]
            interpolation_func_ct = interp1d(subset['J'], subset['Ct'], fill_value="extrapolate")
            interpolation_func_cp = interp1d(subset['J'], subset['Cp'], fill_value="extrapolate")
            
            ct_values.append(interpolation_func_ct(j))
            cp_values.append(interpolation_func_cp(j))
        
        # Calculate the mean, but skip NaN values
        averaged_ct = np.nanmean(ct_values)
        averaged_cp = np.nanmean(cp_values)
        
        averaged_data['Ct'].append(averaged_ct)
        averaged_data['Cp'].append(averaged_cp)
    
    averaged_df = pd.DataFrame(averaged_data)
    
    return averaged_df

def plot_all_ct_vs_j(df: pd.DataFrame, averaged_df: pd.DataFrame, title: str, output_file: str):
    """
    Plots all Ct versus J curves for each PROP RPM, along with the averaged Ct curve.
    
    Args:
        df (pd.DataFrame): The original DataFrame containing all the data.
        averaged_df (pd.DataFrame): The DataFrame containing the averaged data.
        title (str): The title of the plot.
        output_file (str): The path to save the output plot.
    """
    plt.figure(figsize=(10, 6))
    
    rpm_values = df['PROP RPM'].unique()
    
    for rpm in rpm_values:
        subset = df[df['PROP RPM'] == rpm]
        plt.plot(subset['J'], subset['Ct'], label=f'Ct for RPM = {rpm}', alpha=0.6)
    
    # Plot the averaged Ct curve
    plt.plot(averaged_df['J'], averaged_df['Ct'], label='Averaged Ct', color='black', linewidth=2)
    
    plt.xlabel('J (Advance Ratio)')
    plt.ylabel('Ct (Thrust Coefficient)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)

def plot_all_cp_vs_j(df: pd.DataFrame, averaged_df: pd.DataFrame, title: str, output_file: str):
    """
    Plots all Cp versus J curves for each PROP RPM, along with the averaged Cp curve.
    
    Args:
        df (pd.DataFrame): The original DataFrame containing all the data.
        averaged_df (pd.DataFrame): The DataFrame containing the averaged data.
        title (str): The title of the plot.
        output_file (str): The path to save the output plot.
    """
    plt.figure(figsize=(10, 6))
    
    rpm_values = df['PROP RPM'].unique()
    
    for rpm in rpm_values:
        subset = df[df['PROP RPM'] == rpm]
        plt.plot(subset['J'], subset['Cp'], label=f'Cp for RPM = {rpm}', alpha=0.6)
    
    # Plot the averaged Cp curve
    plt.plot(averaged_df['J'], averaged_df['Cp'], label='Averaged Cp', color='black', linewidth=2)
    
    plt.xlabel('J (Advance Ratio)')
    plt.ylabel('Cp (Power Coefficient)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)

def generate_jsb_propeller_xml(df: pd.DataFrame, prop_name: str, output_file: str, geometry_info: dict):
    """
    Generates a JSB propeller XML file from the averaged data.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the averaged data.
        prop_name (str): The name of the propeller.
        output_file (str): The path where the output XML file will be saved.
        geometry_info (dict): A dictionary containing the moment of inertia, radius, and number of blades.
    """
    propeller = ET.Element('propeller', {'name': prop_name, 'version': '1.1'})

    ixx = ET.SubElement(propeller, 'ixx', {'unit': 'KG*M2'})
    ixx.text = str(geometry_info['inertia'])

    diameter = ET.SubElement(propeller, 'diameter', {'unit': 'IN'})
    diameter.text = str(geometry_info['radius'] * 2)

    numblades = ET.SubElement(propeller, 'numblades')
    numblades.text = str(geometry_info['num_blades'])

    constspeed = ET.SubElement(propeller, 'constspeed')
    constspeed.text = '0'

    # Add C_THRUST table
    ct_table = ET.SubElement(propeller, 'table', {'name': 'C_THRUST', 'type': 'internal'})
    ct_data = ET.SubElement(ct_table, 'tableData')
    
    # Manually add indentation and new lines
    ct_data.text = '\n' + ''.join(f'      {row.J:.4f}\t{row.Ct:.4f}\n' for _, row in df.iterrows()) + '    '

    # Add C_POWER table
    cp_table = ET.SubElement(propeller, 'table', {'name': 'C_POWER', 'type': 'internal'})
    cp_data = ET.SubElement(cp_table, 'tableData')
    
    # Manually add indentation and new lines
    cp_data.text = '\n' + ''.join(f'      {row.J:.4f}\t{row.Cp:.4f}\n' for _, row in df.iterrows()) + '    '

    # Generate the tree and write to file
    tree = ET.ElementTree(propeller)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

    # Pretty-print the XML using minidom (for indentation)
    import xml.dom.minidom as minidom
    xml_str = ET.tostring(propeller, 'utf-8')
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

    with open(output_file, 'w') as f:
        f.write(pretty_xml)

def format_xml_with_xmllint(file_path: str):
    """
    Formats the XML file using xmllint command-line tool.
    
    Args:
        file_path (str): The path to the XML file.
    """
    subprocess.run(['xmllint', '--format', file_path, '-o', file_path])

def extract_prefix_and_generate_output_filename(input_filename: str, suffix: str) -> str:
    """
    Extracts the prefix before 'PERF.PE0' in the filename and generates an output file name.
    
    Args:
        input_filename (str): The input filename containing 'PERF.PE0'.
        suffix (str): The suffix to append to the extracted prefix for the output file name.
    
    Returns:
        str: The generated output file name.
    """
    # Extract the base name of the file (in case it's a full path)
    base_name = os.path.basename(input_filename)
    
    # Split the filename at '-PERF.PE0' and take the first part
    prefix = base_name.split('.dat')[0]
    
    # Create the output filename with the desired suffix
    output_filename = f"{prefix}_{suffix}.xml"
    
    return output_filename

def extract_geometry_info(geometry_file: str) -> dict:
    """
    Extracts the moment of inertia, radius, and number of blades from the geometry file.

    Args:
        geometry_file (str): The path to the geometry file.

    Returns:
        dict: A dictionary containing the moment of inertia in Kg-M**2, radius in inches, and the number of blades.
    """
    geometry_info = {
        'inertia': None,
        'radius': None,
        'num_blades': None
    }
    
    with open(geometry_file, 'r') as file:
        for line in file:
            line = line.strip()
            if 'MOMENT OF INERTIA (Kg-M**2)' in line:
                geometry_info['inertia'] = float(line.split('=')[1].strip())
            elif 'RADIUS:' in line:
                geometry_info['radius'] = float(line.split(':')[1].strip().split()[0])
            elif 'BLADES:' in line:
                geometry_info['num_blades'] = int(line.split(':')[1].strip().replace('NUMBER OF BLADES',''))
            
            # Break early if all values are found
            if all(value is not None for value in geometry_info.values()):
                break
    
    # Validate that all required information has been extracted
    if any(value is None for value in geometry_info.values()):
        raise ValueError(f"One or more required values not found in {geometry_file}")
    
    return geometry_info

def process_propeller_file(propeller_file: str, output_folder: str):
    """
    Processes a single propeller data file to generate the JSB propeller XML.
    
    Args:
        propeller_file (str): The path to the propeller data file.
    """
    base_name = os.path.basename(propeller_file)
    prop_name = base_name.split('.dat')[0].replace('PER3_', '')

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_filename = f"{output_folder}/{prop_name}.xml"

    df = parse_data(propeller_file)
    averaged_df = interpolate_and_average(df)
    

    plot_all_ct_vs_j(df, averaged_df, f"All Ct vs. J for {prop_name}", f"{output_folder}/{prop_name}_all_ct.png")
    plot_all_cp_vs_j(df, averaged_df, f"All Cp vs. J for {prop_name}", f"{output_folder}/{prop_name}_all_cp.png")
    
    geometry_file = f"geometry/{prop_name}-PERF.PE0"
    geometry_info = extract_geometry_info(geometry_file)

    generate_jsb_propeller_xml(averaged_df, prop_name, output_filename, geometry_info)

def main():
    parser = argparse.ArgumentParser(description="Plot averaged Ct vs. J and generate JSB propeller XML model.")
    parser.add_argument('propeller_path', type=str, help='The path to the propeller data file or directory containing multiple propeller data files.')
    parser.add_argument('--output_folder', type=str, default='jsb_apc_propeller_models', help='The folder where the output files will be saved.')
    args = parser.parse_args()

    # Check if the provided path is a directory or a file
    if os.path.isdir(args.propeller_path):
        # Iterate through all .dat files in the directory
        for filename in os.listdir(args.propeller_path):
            if filename.endswith('.dat'):
                propeller_file = os.path.join(args.propeller_path, filename)
                process_propeller_file(propeller_file, args.output_folder)
    else:
        # Process a single file
        process_propeller_file(args.propeller_path, args.output_folder)

if __name__ == "__main__":
    main()
