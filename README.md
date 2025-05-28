# VAE

VAE (Variational Autoencoder) implementation in PyTorch. 

## Results - train
| Title | Epoch 0 | Epoch 1 | Epoch 10 | Epoch 30 | Animation |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Generated | ![generated_0](https://github.com/ShotaDeguchi/Variational_Autoencoder/assets/49257696/f8467a4c-5728-499f-bc61-832f452f841e) | ![generated_1](https://github.com/ShotaDeguchi/Variational_Autoencoder/assets/49257696/38ada7aa-c0e2-4d21-b3d8-b5de7a5db745) | ![generated_10](https://github.com/ShotaDeguchi/Variational_Autoencoder/assets/49257696/671a40f7-e589-4358-ad5a-ba3c7f40d6ee) | ![generated_30](https://github.com/ShotaDeguchi/Variational_Autoencoder/assets/49257696/3bdbfafd-0d71-4d22-8b97-b2466a686f3d) | ![generated_mnist](https://github.com/user-attachments/assets/dfdbfd0b-d282-48e2-896c-0855d7533127) |
| 2D latent space | ![embedding_0](https://github.com/ShotaDeguchi/Variational_Autoencoder/assets/49257696/0421e966-59e6-4180-85fc-7c3b411f26c2) | ![embedding_1](https://github.com/ShotaDeguchi/Variational_Autoencoder/assets/49257696/d95b0e71-6a93-485a-a83e-812a79f663b8) | ![embedding_10](https://github.com/ShotaDeguchi/Variational_Autoencoder/assets/49257696/8f23b0e9-8e49-426a-880c-e288cd204028) | ![embedding_30](https://github.com/ShotaDeguchi/Variational_Autoencoder/assets/49257696/c83653a6-bd68-4710-8efb-22f5f495548f) | ![2D_embedding_mnist](https://github.com/user-attachments/assets/ac36ce83-73e2-43f1-b73e-0ca38a16c856) |

## Results - infer
| Sweep along z1 | Sweep along z2 | Walk from (-2, 2) to (2, -2) |
| :---: | :---: | :---: |
| ![ezgif com-gif-maker (2)](https://github.com/ShotaDeguchi/Variational_Autoencoder/assets/49257696/dad38b18-412e-4ea5-aca0-7ff95b3b9ba6) | ![ezgif com-gif-maker (3)](https://github.com/ShotaDeguchi/Variational_Autoencoder/assets/49257696/5db5c4cb-5f34-4dda-8478-d8248b6c3d80) | ![ezgif com-gif-maker (4)](https://github.com/ShotaDeguchi/Variational_Autoencoder/assets/49257696/cf827a43-8869-4f77-9a5b-96a735c740b7) |
