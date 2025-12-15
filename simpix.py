import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import math
import argparse

# Try to import numba
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not found. Execution will be slow.")

def color_diff_sq(c1, c2):
    """Calculates squared Euclidean distance between two colors."""
    # Using simple Euclidean distance in RGB space as per common implementations
    # The C++ code uses a weighted version, but pure Euclidean is often sufficient and faster
    dr = int(c1[0]) - int(c2[0])
    dg = int(c1[1]) - int(c2[1])
    db = int(c1[2]) - int(c2[2])
    # Weighted Euclidean from Wikipedia (and C++ code) for better perceptual uniformity
    # rmean = (c1[0] + c2[0]) / 2
    # if rmean < 128:
    #     return 2*dr*dr + 4*dg*dg + 3*db*db
    # else:
    #     return 3*dr*dr + 4*dg*dg + 2*db*db
    return dr*dr + dg*dg + db*db

if HAS_NUMBA:
    @njit(fastmath=True)
    def energy_pixel(p1, p2):
        rmean = (p1[0] + p2[0]) * 0.5
        dr = p1[0] - p2[0]
        dg = p1[1] - p2[1]
        db = p1[2] - p2[2]
        wt_r = 2.0 + rmean / 256.0
        wt_b = 2.0 + (255.0 - rmean) / 256.0
        return wt_r * dr*dr + 4.0 * dg*dg + wt_b * db*db

    @njit(fastmath=True)
    def calculate_total_energy(current_pixels, target_pixels):
        total_e = 0.0
        n = current_pixels.shape[0]
        for i in range(n):
            total_e += math.sqrt(energy_pixel(current_pixels[i], target_pixels[i]))
        return total_e

    @njit(fastmath=True)
    def anneal_step(current_pixels, target_pixels, temp, n_steps):
        n_pixels = current_pixels.shape[0]
        accepted_moves = 0
        current_energy_change = 0.0
        
        for _ in range(n_steps):
            move_type = 0
            if np.random.random() < 0.2:
                move_type = 1
            
            if move_type == 0: # SWAP
                i = np.random.randint(0, n_pixels)
                j = np.random.randint(0, n_pixels)
                if i == j: continue

                p_i = current_pixels[i]
                p_j = current_pixels[j]
                
                t_i = target_pixels[i]
                t_j = target_pixels[j]
                
                e_old = math.sqrt(energy_pixel(p_i, t_i)) + math.sqrt(energy_pixel(p_j, t_j))
                e_new = math.sqrt(energy_pixel(p_j, t_i)) + math.sqrt(energy_pixel(p_i, t_j))
                
                delta_E = e_new - e_old
                
                if delta_E < 0 or np.random.random() < math.exp(-delta_E / temp):
                   
                    tmp_r, tmp_g, tmp_b = p_i[0], p_i[1], p_i[2]
                    current_pixels[i, 0], current_pixels[i, 1], current_pixels[i, 2] = p_j[0], p_j[1], p_j[2]
                    current_pixels[j, 0], current_pixels[j, 1], current_pixels[j, 2] = tmp_r, tmp_g, tmp_b
                    accepted_moves += 1
                    current_energy_change += delta_E

            else:
                seg_len = np.random.randint(2, min(200, n_pixels // 10)) 
                start_idx = np.random.randint(0, n_pixels - seg_len)
                end_idx = start_idx + seg_len - 1
                
                
                delta_E = 0.0
                
               
                for k in range(seg_len):
                    idx_fwd = start_idx + k
                    idx_rev = end_idx - k
                    
                    p_orig = current_pixels[idx_fwd] 
                    p_swap = current_pixels[idx_rev] 
                    
                    t_fwd = target_pixels[idx_fwd]
                    
                    
                    e_old_k = math.sqrt(energy_pixel(p_orig, t_fwd))
                    e_new_k = math.sqrt(energy_pixel(p_swap, t_fwd))
                    
                    delta_E += (e_new_k - e_old_k)
                    
                if delta_E < 0 or np.random.random() < math.exp(-delta_E / temp):
                   
                    half_len = seg_len // 2
                    for k in range(half_len):
                        i = start_idx + k
                        j = end_idx - k
                        
                        p_i = current_pixels[i]
                        p_j = current_pixels[j]
                        
                        tmp_r, tmp_g, tmp_b = p_i[0], p_i[1], p_i[2]
                        current_pixels[i, 0], current_pixels[i, 1], current_pixels[i, 2] = p_j[0], p_j[1], p_j[2]
                        current_pixels[j, 0], current_pixels[j, 1], current_pixels[j, 2] = tmp_r, tmp_g, tmp_b
                        
                    accepted_moves += 1
                    current_energy_change += delta_E

        return accepted_moves, current_energy_change

def main():
    parser = argparse.ArgumentParser(description="SimPix: Reconstruct image using pixels from another via Simulated Annealing")
    parser.add_argument("source", help="Path to source image")
    parser.add_argument("target", help="Path to target image")
    parser.add_argument("--output", default="output.png", help="Path to output image")
    parser.add_argument("--steps", type=int, default=2000000, help="Total annealing steps")
    parser.add_argument("--temp_start", type=float, default=1000.0, help="Initial temperature")
    parser.add_argument("--temp_end", type=float, default=0.1, help="Final temperature")
    
    args = parser.parse_args()
    
    # 1. Load Images
    print(f"Loading images: {args.source} -> {args.target}")
    try:
        src_img = Image.open(args.source).convert("RGB")
        tgt_img = Image.open(args.target).convert("RGB")
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # 2. Validation
    if src_img.size != tgt_img.size:
        print(f"Dimensions match check: {src_img.size} vs {tgt_img.size}")
        print("Resizing target to match source...")
        tgt_img = tgt_img.resize(src_img.size)

    width, height = src_img.size
    print(f"Processing image size: {width}x{height} ({width*height} pixels)")

    # 3. Prepare Data
    src_arr = np.array(src_img, dtype=np.float32)
    tgt_arr = np.array(tgt_img, dtype=np.float32)
    
    # Flatten for easier manipulation
    flat_src = src_arr.reshape(-1, 3)
    flat_tgt = tgt_arr.reshape(-1, 3)
    
    # Initial "Hot" Scramble (as per reference C++ code)
    print("Initializing random state...")
    np.random.shuffle(flat_src) # Start with random permutation of source pixels
    
    # 4. Annealing Loop
    current_pixels = flat_src.copy() # Working copy
    target_pixels = flat_tgt
    
    start_time = time.time()
    
    # Annealing Schedule
    steps_total = args.steps
    # We'll break it down into epochs for reporting
    epochs = 100
    steps_per_epoch = steps_total // epochs
    
    # Exponential cooling schedule parameters
    # T(k) = T_start * (T_end / T_start) ^ (k / k_max)
    # or simple multiplicative decay factor alpha such that T_end = T_start * alpha^epochs
    alpha = (args.temp_end / args.temp_start) ** (1.0 / epochs)
    temperature = args.temp_start
    
    initial_energy = calculate_total_energy(current_pixels, target_pixels)
    print(f"Initial Energy: {initial_energy:.2f}")

    print(f"Starting Annealing ({steps_total} steps, T={args.temp_start} -> {args.temp_end})...")
    
    current_energy = initial_energy
    
    bs_start = time.time()
    for e in range(epochs):
        # Run a batch of steps
        # n_steps argument to numba function avoids python loop overhead
        acc, delta_E = anneal_step(current_pixels, target_pixels, temperature, steps_per_epoch)
        
        # update temperature
        temperature *= alpha
        
        if (e+1) % 10 == 0:
             current_energy = calculate_total_energy(current_pixels, target_pixels)
        else:
             pass
             
        # Progress report
        elapsed = time.time() - bs_start
        print(f"Epoch {e+1}/{epochs}: T={temperature:.2f}, Energy={current_energy:.0f} (Time: {elapsed:.1f}s)")

    end_time = time.time()
    final_energy = calculate_total_energy(current_pixels, target_pixels)
    print(f"Done in {end_time - start_time:.2f}s. Final Energy: {final_energy:.2f}")
    
    # 5. Reconstruct and Save
    out_arr = current_pixels.reshape(height, width, 3).astype(np.uint8)
    out_img = Image.fromarray(out_arr)
    out_img.save(args.output)
    print(f"Saved output to {args.output}")

    # 6. Generate Collage
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(src_img)
    axs[0].set_title("Source (Pixels Used)")
    axs[0].axis('off')

    axs[1].imshow(tgt_img)
    axs[1].set_title("Target (Goal)")
    axs[1].axis('off')

    axs[2].imshow(out_img)
    axs[2].set_title("Reconstruction")
    axs[2].axis('off')

    plt.tight_layout()
    collage_path = "collage.png"
    plt.savefig(collage_path)
    print(f"Saved collage to {collage_path}")
    # plt.show() # Don't block if running in batch

if __name__ == "__main__":
    if not HAS_NUMBA:
        print("CRITICAL: Numba is needed for performance. Install valid environment.")
    main()
