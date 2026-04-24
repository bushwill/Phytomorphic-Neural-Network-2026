import os
import sys
import shutil
import subprocess
import datetime

# Set umask to 0 so that files created inside docker are accessible on host
os.umask(0)

def main():
    # Base L-system directory (dynamic to work inside and outside Docker)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lsystem_dir = os.path.join(base_dir, "lsystem")
    
    # Target output directory for images
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, "l-system images", f"run_{timestamp}")
    print(f"Creating execution directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Typical baseline values for the L-system parameters
    params = {
        "MAX_PHYTOMERS": 10.0,
        "PLASTOCHRON": 3.0,
        "PlantRollAng": 45.0,  # Rotated 45 degrees to give a clear 3D/isometric perspective
        "PlantDownAng": 0.0,
        "BrAngle": 135.0,
        "LeafLen": 5.0,
        "ExpLeafWid": 0.5,
        "LeafWid": 1.0,
        "LEAF_BEND_SCALE": 90.0,
        "LEAF_TWIST_SCALE": 180.0,
        "IntLen": 0.7,
        "IntWid": 0.9,
        "ExpIntRad": 0.5
    }

    # You can change these parameters if you want to test specific bounded extremes (e.g. plastochron 2.8 vs 3.2)
    
    # 1. Create a temporary parameter file for the L-system
    param_file = os.path.join(base_dir, "temp_anim_params.vset")
    with open(param_file, "w") as f:
        for key, value in params.items():
            f.write(f"#define {key} {value}\n")

    print(f"Created temporary parameter file at {param_file}")

    # Helper to resolve lsystem file paths
    def ls(f): 
        return os.path.join(lsystem_dir, f)

    # 2. Build the lpfg command for animated execution
    # Running lpfg -a anim.a will trigger the L-system's OutputFrame() function
    lpfg_args = [
        "lpfg",
        "-w", "1024", "1024", # Output resolution bumped to 1024 for clarity
        "-a", ls("anim.a"), # Animation trigger
        ls("lsystem.l"),
        ls("view.v"),
        ls("materials.mat"),
        ls("contours.cset"),
        ls("functions.fset"),
        ls("functions.tset"),
        param_file
    ]

    print(f"Running lpfg animation inside: {output_dir}")
    print(f"Command run: {' '.join(lpfg_args)}")
    
    # 3. Execute lpfg inside the output_dir so any generated maize*.png 
    # and leafnumber.csv files are placed directly into `l-system images`
    process = subprocess.Popen(lpfg_args, cwd=output_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, _ = process.communicate()
    
    if process.returncode != 0:
        print("\n=== L-system execution failed! ===")
        print(stdout.decode('utf-8'))
        sys.exit(1)
        
    print("\nAnimation successfully generated!")
    
    # Cleanup temporary parameter
    if os.path.exists(param_file):
        os.remove(param_file)
        
    # List the generated files
    generated_files = sorted(os.listdir(output_dir))
    print(f"\nImages saved in '{output_dir}':")
    for f in generated_files:
        if f.endswith('.png'):
            print(f"  - {f}")
            
if __name__ == '__main__':
    main()