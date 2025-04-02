Text2Sketch2Face
==============================

This Flask-based application generates sketches from textual descriptions using deep learning and generative models. It also supports transforming sketches into realistic facial images using advanced image translation techniques.

Key Features:

Text-to-Sketch Generation:

Uses Multi-Head Self-Attention for text encoding.

Employs a Conditional Variational Autoencoder (CVAE) conditioned on extracted attributes for generating sketches.

Refines sketches using a Conditional Generative Adversarial Network (CGAN) to enhance high-fidelity facial details.

Sketch-to-Face Translation:

Evaluates multiple image translation techniques, including supervised models, CycleGAN, and Pix2Pix.

Achieves the best results with CycleGAN, producing highly realistic facial images from sketches.
