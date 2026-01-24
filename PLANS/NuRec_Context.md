Building realistic 3D environments for robotics simulation can be a labor-intensive process. Now, with NVIDIA Omniverse NuRec, you can complete the entire process using just a smartphone. This post walks you through each step—from capturing photos using an iPhone to rebuilding the scene in 3D using 3DGUT to loading it into NVIDIA Isaac Sim and inserting a robot. To skip the reconstruction (Steps 1-3) and explore this scene directly in Isaac Sim (Step 4), visit NVIDIA Physical AI on Hugging Face.

Step 1: Capture the real-world scene 
The first step is to take photos of the real environment to be reconstructed. This no longer requires special hardware—you can use a regular smartphone camera. 

As you walk around your scene, take photos using your phone. A few photogrammetry best practices are outlined below:

Lighting and focus should be right and steady. Avoid fast motion and blur. If you can, use a faster shutter speed (for example, 1/100 second or faster)

In the built-in camera app, you can’t directly set shutter speed, but you can:
Lock focus/exposure: Long-press on the subject to enable AE/AF Lock, then drag the exposure slider down slightly (−0.3 to −0.7 EV) to keep highlights crisp.
Stabilize: Use a tripod or lean against a wall; the sharper each frame, the better COLMAP tracks features. 
Avoid auto macro switching: To ensure that focal length doesn’t jump shot-to-shot on iPhone Pro models, turn off Settings → Camera → Auto Macro. 
For manual shutter/ISO control, try any of the following iOS apps that allow fixed shutter plus ISO: Lumina: Manual Camera, Halide, ProCamera, ProCam 8, Moment Pro Camera, Lightroom Mobile.
Start with 1/120–1/250 seconds outdoors and ≥1/100 seconds indoors. Set ISO as low as possible while keeping exposure acceptable.
Lock white balance to avoid color shifts between frames.  
For coverage, make a slow loop around the area and capture multiple heights and angles. A safe rule of thumb is 60% overlap.

Tip: COLMAP expects standard image formats. If your iPhone saves HEIC, either switch to Settings → Camera → Formats → Most Compatible (shoots JPEG), or export/convert to JPG before COLMAP.  

Step 2: Generate a sparse reconstruction with COLMAP
When you have your photos, the next step is to figure out the 3D structure and camera positions from those images. This example uses COLMAP, a popular open source Structure-from-Motion (SfM) and Multi-View Stereo pipeline, to do this. COLMAP will create a sparse point cloud of the scene and estimate the camera parameters for each photo. 

COLMAP also provides the following command-line path. However, note that the GUI Automatic Reconstruction is the easier way to start. For compatibility with 3DGUT, select either the pinhole or simple pinhole camera model.

# Feature detection & extraction
$ colmap feature_extractor \
    --database_path ./colmap/database.db \
    --image_path    ./images/ \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model   PINHOLE \
    --SiftExtraction.max_image_size 2000 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1
 
# Feature matching
$ colmap exhaustive_matcher \
    --database_path ./colmap/database.db \
    --SiftMatching.use_gpu 1
 
# Global SFM
$ colmap mapper \
    --database_path ./colmap/database.db \
    --image_path    ./images/ \
    --output_path   ./colmap/sparse
 
# Visualize for verification
$ colmap gui --import_path ./colmap/sparse/0 \
    --database_path ./colmap/database.db \
    --image_path    ./images/
COLMAP will output a project folder (often containing a database.db, an images/ folder, and a sparse/ directory with the reconstruction data). Once COLMAP completes, you’ll have: 

A sparse point cloud of the scene
Camera pose data for all your images
This is the information needed to feed into the 3D reconstruction with 3DGUT (Step 3).

Step 3: Train a dense 3D reconstruction with 3DGUT and export to USD
Now for the magic. The next step uses the 3DGUT algorithm to turn the sparse model and images into a dense, photorealistic 3D scene:

Set up the 3DGUT environment: The 3DGUT repository requires a Linux system with CUDA 11.8, GCC ≤ 11 and an NVIDIA GPU. To see the official 3DGUT code, visit nv-tlabs/3dgrut on GitHub and follow the instructions to install the required libraries. 
Clone the 3DGUT repo: Using the following command, clone and install the 3DGUT repo. To verify that the clone is successful before proceeding, run a test reconstruction on one of the sample datasets in the repo to ensure it’s working.
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut
chmod +x install_env.sh
./install_env.sh 3dgrut
conda activate 3dgrut
Prepare the COLMAP outputs: Ensure you know the path to your COLMAP output directory generated in Step 2. This example uses the apps/colmap_3dgut_mcmc.yaml config because it pairs 3DGUT with an MCMC (Markov Chain Monte Carlo) densification strategy. In practice, this samples and densifies Gaussians where the reconstruction is uncertain, sharpening thin structures and edges and improving overall fidelity with only a modest training time overhead compared to the baseline config.
Run the 3DGUT training script and export USDZ: With the environment active, you can start training by running the provided train.py script with the COLMAP config. For example, the command might look like similar to the following:
$ conda activate 3dgrut
$ python train.py --config-name apps/colmap_3dgut_mcmc.yaml \
    path=/path/to/colmap/ \
    out_dir=/path/to/out/ \
    experiment_name=3dgut_mcmc \
    export_usdz.enabled=true \
    export_usdz.apply_normalizing_transform=true
Once you run the command, 3DGUT will begin training. It will read in your images and COLMAP data and start optimizing a 3D representation. This may take some time depending on your scene size and GPU—anywhere from a few minutes for a small scene to a few hours for a very detailed one.

When the process completes, you should have a dense reconstruction of your scene. The output includes a directory with model checkpoints. Setting the flags export_usdz.enabled=true and export_usdz.apply_normalizaing_transform=true also generates a USDZ file. 

export_usdz.enabled=true writes a USDZ of your reconstructed scene, so you can load it straight into Isaac Sim.
export_usdz.apply_normalizing_transform=true applies a primitive normalization (centers/scales the scene near the origin). It does not guarantee the floor is exactly at z = 0. In Isaac Sim, you can add a Ground Plane or nudge the scene root (translate/rotate) for alignment.
Exporting the reconstructed scene to USD makes it plug-and-play with Isaac Sim. The produced USDZ file uses a custom USD schema and essentially serves as a packaged USD scene containing all the Gaussian-splatting data of the 3D reconstruction. Note that a standardized schema is under discussion within AOUSD.  

Step 4: Deploy the reconstructed scene in Isaac Sim and add a robot
Now, for the fun part. With the real world scene now fully reconstructed in USD, it can be used in Isaac Sim to train and test virtual robots.  

To load your scene and insert a robot, follow these steps:

1. Launch Isaac Sim (version 5.0 or later, which supports the NuRec/3DGUT features). When Isaac Sim is open, start with an empty stage (File > New).

2. Import your USD scene: From the menu, navigate to File > Import and find the USDZ file of your scene. Alternatively, drag-and-drop the USDZ file from the Isaac Sim content browser into the scene. When Isaac Sim loads the file, you should see your reconstructed environment appear in the viewport as a collection of colorful points, or Gaussians. It may look almost like a photoreal 3D photo when viewed through the camera.

Tip: You can use the Isaac Sim navigation (WASD keys or right-click and drag-and-drop) to fly around the scene and inspect it from different angles.
3. Add a ground plane for physics: Your reconstructed scene is simply visual geometry (the points/voxels from 3DGUT) with no inherent collision properties. To have a robot move around, add a ground plane so the robot has something to stand on. In Isaac Sim, this is easy: click Create > Physics > Ground Plane. A flat ground will appear (usually at z = 0) covering your scene’s floor area. You may want to scale it (for example, x to 100, y to 100, as shown in Video 1). Adjust Translate / Rotate / Scale so it aligns to the reconstructed floor.

Next, connect a proxy mesh to receive shadows. A proxy mesh is a simple stand-in that grounds objects visually in the scene by providing a surface to cast shadows onto. You’ve already created a mesh, which is your ground plane. To connect it as a proxy, follow these steps:

Select the NuRec prim (the volume prim under the global xform). In the Raw USD Properties panel, locate the NuRec/Volume section and find the Proxy field.
Click + to add your proxy mesh prim.
Select your ground plane again. Ensure that the Geometry > Matte Object property is turned on.

Video 1. Learn how to add a ground plane mesh and physics to the rendered scene while importing into Isaac Sim
4. Insert a robot from Isaac Sim assets: Isaac Sim includes many SimReady robot models. To add one to the scene, navigate to the top menu bar and select Create > Robots. Choose a robot—the popular Franka Emika Panda robotic arm, for example. You could also choose a mobile robot such as LeatherBack/Carter/TurtleBot, or even a humanoid, depending on what’s available in your Isaac Sim library. 

When you click the robot asset of your choice, it will be added to your scene. Use the Move/Rotate tools to position the robot where you want it in your reconstructed scene. 

5. Press Play and watch: At this point, the robot should be sitting or standing in your photoreal 3D scene. If it’s a manipulator arm, you can animate it or even run reinforcement learning—depending on your use-case. Now that the real-world environment is in Isaac Sim, you can treat it like any other simulation environment.

Get started reconstructing a scene in NVIDIA Isaac Sim 
This simple real-to-sim workflow turns everyday iPhone photos into an interactive, robot-ready scene. It’s as easy as capture → COLMAP → 3DGUT reconstruction → USDZ export → load in NVIDIA Isaac Sim. The workflow replaces long photogrammetry loops with a simple, reproducible path to digital twins that you can drive, plan, and test quickly.
