Text2Sketch2Face
==============================

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text-to-Sketch & Sketch-to-Face Generation</title>
</head>
<body>
    <p>This Flask-based application generates sketches from textual descriptions using deep learning and generative models. It also supports transforming sketches into realistic facial images using advanced image translation techniques.</p>

    <h2>Key Features:</h2>
    
    <h3>Text-to-Sketch Generation:</h3>
    <ul>
        <li>Uses <strong>Multi-Head Self-Attention</strong> for text encoding.</li>
        <li>Employs a <strong>Conditional Variational Autoencoder (CVAE)</strong> conditioned on extracted attributes for generating sketches.</li>
        <li>Refines sketches using a <strong>Conditional Generative Adversarial Network (CGAN)</strong> to enhance high-fidelity facial details.</li>
    </ul>

    <h3>Sketch-to-Face Translation:</h3>
    <ul>
        <li>Evaluates multiple image translation techniques, including <strong>supervised models, CycleGAN, and Pix2Pix</strong>.</li>
        <li>Achieves the best results with <strong>CycleGAN</strong>, producing highly realistic facial images from sketches.</li>
    </ul>

    <p>This project combines state-of-the-art generative techniques to bridge the gap between textual descriptions, sketches, and photorealistic facial reconstructions.</p>
</body>
</html>

