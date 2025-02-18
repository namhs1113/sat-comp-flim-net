# SatCompFLImNet (Deep Learning-based Saturation Compensation for High Dynamic Range Multispectral Fluorescence Lifetime Imaging)
In multispectral fluorescence lifetime imaging (FLIm), achieving consistent imaging quality across all spectral channels is crucial for accurately identifying a wide range of fluorophores. However, these essential measurements are frequently compromised by saturation artifacts due to the inherently limited dynamic range of detection systems. To address this issue, we present SatCompFLImNet, a deep learning-based network specifically designed to correct saturation artifacts in multispectral FLIm, facilitating high dynamic range applications. Leveraging generative adversarial networks, SatCompFLImNet effectively compensates for saturated fluorescence signals, ensuring accurate lifetime measurements across various levels of saturation. Extensively validated with simulated and real-world data, SatCompFLImNet demonstrates remarkable capability in correcting saturation artifacts, improving signal-to-noise ratios, and maintaining fidelity of lifetime measurements. By ensuring reliable fluorescence lifetime measurements under a variety of saturation conditions, SatCompFLImNet paves the way for improved diagnostic tools and a deeper understanding of biological processes, making it a pivotal advancement for research and clinical diagnostics in tissue characterization and disease pathogenesis.

# Dataset Download Instructions
To download the dataset, run the following commands in your terminal:

```bash
curl -L -o dataset.zip "https://www.dropbox.com/scl/fi/qfwo3257dmubn17506nv3/dataset.zip?rlkey=ys6q8lo5x8rqblhrv49ojcyd5&st=cfs6bb7d&dl=1"
unzip dataset.zip
