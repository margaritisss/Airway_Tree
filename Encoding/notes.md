DATA AUGMENTATION

"The data is augmented by adding random translations and horizontal flips to each training example, as in [5]"
"Then training on one noisy and one uncorrupted copy of each instance, randomly shuffled"

το σκεπτικο
The rationale: "By training the network to reconstruct both corrupted and uncorrupted data, we force it to learn invariance to small structural variations."

SEARCH

Should I add validation dataset ?

DISAGREEMENTS with the paper and the code

γ: paper says 0.97, released code uses 0.98. Cannot satisfy both.
Output clip range: paper says [0.1, 1), released code uses [1e-7, 1-1e-7]. Cannot satisfy both.



A subtle note on the BCE reduction
Both versions use mean() over all elements (B × 1 × 128 × 128 × 128 in your case, B × 1 × 32 × 32 × 32 in theirs). This means the recon loss is on a per-voxel scale, so it's automatically comparable across input resolutions.
But the KL is also mean() — over B × num_latents. So you're comparing two means at very different "scales" of the underlying data. With your 128³ inputs and 100 latents:

Recon mean is over B × 1 × 128³ ≈ B × 2.1M elements
KL mean is over B × 100 elements

This means the relative weighting between recon and KL is the same as in their 32³ setup (because both use mean), but the implicit β coefficient is built into the mean reductions and matches what they did. So that's fine — but worth knowing if you ever want to rebalance with an explicit β.