import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import subprocess

def plot_images(images, nrows, ncols, figsize, cmap='viridis'):
	num_images = len(images)
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=100)

	for i in range(nrows):
		for j in range(ncols):
			index = i * ncols + j
			if index < num_images:
				ax = axes[i, j]
				ax.imshow(images[index], cmap=cmap, aspect='auto')
				ax.set_title(f'Image {index+1}')
			ax.axis('off') # Turns off axis for empty plots

	temp_image = os.path.join(tempfile.gettempdir(), 'temp_heatmap.png')
	plt.savefig(temp_image, bbox_inches='tight')
	plt.close()
	subprocess.run("imv "+temp_image, shell=True, check=True, text=True, capture_output=True)