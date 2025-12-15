# simpix

C++ starter code
* simpix_start.cpp
use make to build this example

Usage: simapix_start image1 image2 <output=out.png>

Python starter code
* simpix_start.py

Usage: simapix_start image1 image2 <output=out.png>



## Files
- `simpix.py`: Main Python script
- `download_and_prepare.py`: Script to download/prepare images.
- `simpix_files/`: Directory containing input and output images.

## Build and Run

### Requirements
- NumPy, Pillow (PIL), Matplotlib, Numba

Install dependencies:
```bash
pip install numpy pillow matplotlib numba
```

### Quick Start
1.  **Prepare Images**:
    ```bash
    python download_and_prepare.py
    ```

2.  **Run Simulations**:
    *   **Impressionist Pair (Monet -> Renoir)**:
        ```bash
        python simpix.py simpix_files/monet_sunrise_640.png simpix_files/renoir_moulin_640.png --output simpix_files/impressionist_result.png --steps 5000000
        ```
    *   **Local Pair (Frisbee -> Rotunda)**:
        ```bash
        python simpix.py simpix_files/frisbee_640.png simpix_files/rotunda_640.png --output simpix_files/rotunda_result.png --steps 5000000
        ```

## Algorithm Details
The program uses a Metropolis-Hastings algorithm:
1.  **State**: Current arrangement of pixels.
2.  **Energy**: Sum of Euclidean distances (weighted for RGB perception) between current pixels and target pixels.
3.  **Moves**:
    *   **Swap**: Swap two random pixels.
    *   **Reversal**: Reverse a random segment of pixels.
4.  **Cooling**: Exponential decay from high temperature to near-zero.

## Performance
- **Image Size**: 640x480
- **Steps**: 5,000,000

## Results
To view converted images see simpix_files folder
The algorithm successfully transforms the source image structure to match the target while strictly maintaining the source's color histogram.
