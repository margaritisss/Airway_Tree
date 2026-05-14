DATA AUGMENTATION

"The data is augmented by adding random translations and horizontal flips to each training example, as in [5]"
"Then training on one noisy and one uncorrupted copy of each instance, randomly shuffled"

το σκεπτικο
The rationale: "By training the network to reconstruct both corrupted and uncorrupted data, we force it to learn invariance to small structural variations."