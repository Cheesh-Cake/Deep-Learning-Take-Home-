Tugas Take-Home **Deep Learning (D6421)**.
Proyek ini membandingkan enam arsitektur generatif pada dataset dan protokol evaluasi yang sama, serta membuktikan secara empiris mengapa bobot dan fitur laten **MLP Classifier** tidak dapat dipakai untuk image generation.

**Penulis:** `Achmad Sanusi` · **NIM:** `2802515280` · **Kelas:** `LB01`

(https://github.com/Cheesh-Cake/Deep-Learning-Take-Home-)

## Isi Proyek

| Komponen | Keterangan |
|----------|------------|
| Vanilla AE | 784→256→64→256→784, loss MSE |
| Sparse AE | Arsitektur sama, penalti L1 pada hidden layer (λ = 1e-2) |
| VAE | Encoder (μ, log σ²), reparameterisasi, loss BCE + KL |
| Vanilla GAN | Generator dan Discriminator MLP, loss non-saturating |
| DCGAN | Generator ConvTranspose dan Discriminator Conv, BatchNorm, LeakyReLU |
| DDPM | UNet dengan time-embedding, T = 1000, prediksi derau, EMA |
| MLP Classifier | 784→256→64→10 untuk eksperimen pembuktian batasan (Bagian 2 tugas) |

**Eksperimen pembuktian batasan MLP Classifier (5 tahap):** pelatihan classifier (akurasi test ≥ 88%), proyeksi balik transpos bobot dari vektor kelas, proyeksi balik dari latent test image, *frozen latent decoding* dibandingkan Vanilla AE, dan prior sampling pada ruang laten classifier.

**Metrik:** Test MSE, PSNR, SSIM (rekonstruksi), FID (sampel sintetis), training time per epoch, inference latency per batch 64.

## Struktur Repositori

```
.
├── Tugas_DL_Sesi07_Generative_Models.ipynb   # seluruh kode (model, training, evaluasi, figur)
├── buat_laporan_pdf.py                       # membuat laporan PDF dari hasil run
├── requirements.txt
├── README.md
└── outputs/                                  # dihasilkan saat notebook dijalankan
    ├── results.json                          # metrik 6 model
    ├── mlp_limits.json                       # hasil eksperimen MLP Classifier
    └── fig1_reconstruction.png ... fig5_loss_ddpm.png
```

## Cara Menjalankan

### Opsi A: Google Colab (disarankan)
1. Klik badge **Open In Colab** di atas, atau upload notebook ke colab.research.google.com.
2. Aktifkan GPU: *Runtime → Change runtime type → T4 GPU*.
3. Jalankan sel dari atas ke bawah. Di sel **Konfigurasi**, `QUICK = True` untuk uji cepat (beberapa menit); ubah ke `QUICK = False` untuk run penuh.
4. Hasil tersimpan di folder `outputs/` dan terunduh otomatis sebagai `outputs.zip`.

## Protokol Eksperimen (standar untuk semua model)

- Dataset: **Fashion-MNIST** (70.000 citra, 28×28 grayscale, 10 kelas) via `torchvision.datasets`
- Split: 80% train / 10% validation / 10% test (generator acak tetap)
- Batch size 128; `shuffle=True` hanya pada training set
- Seed tetap: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`
- Seluruh metrik dihitung pada **Test Set**
- Konfigurasi run penuh: AE/Sparse AE/VAE 30 epoch, Vanilla GAN 50 epoch, DCGAN 30 epoch, DDPM 30 epoch (T = 1000), MLP Classifier 40 epoch

## Hasil

> Isi tabel dari `outputs/results.json` setelah run penuh.

| Model | Dim Laten | Train (s/epoch) | Latency (ms/batch 64) | MSE | PSNR (dB) | SSIM | FID |
|---|---|---|---|---|---|---|---|
| Vanilla AE | 64 | | | | | | N/A |
| Sparse AE | 64 / 256 | | | | | | N/A |
| VAE | 64 | | | | | | |
| Vanilla GAN | 100 | | | N/A | N/A | N/A | |
| DCGAN | 100 | | | N/A | N/A | N/A | |
| DDPM | 28×28 | | | N/A | N/A | N/A | |

**MLP Classifier:** akurasi test `__%`; frozen decoder `z_MLP` (MSE / PSNR / SSIM) `__` vs Vanilla AE `z_AE` `__`.

### Visualisasi
| | |
|---|---|
| Rekonstruksi | ![fig1](outputs/fig1_reconstruction.png) |
| Sampling generatif | ![fig2](outputs/fig2_sampling.png) |
| Batasan MLP Classifier | ![fig3](outputs/fig3_mlp_limits.png) |
| Interpolasi latent | ![fig4](outputs/fig4_interpolation.png) |
| Kurva loss & trajektori DDPM | ![fig5](outputs/fig5_loss_ddpm.png) |

## Ringkasan Temuan

Isi dengan kesimpulan singkat dari hasil Anda, misalnya: urutan fidelitas rekonstruksi, urutan FID, kestabilan training GAN vs DDPM, dan bukti bahwa `z_MLP` tidak dapat disampel. Uraian lengkap ada pada laporan PDF.

## Referensi

Goodfellow et al. (2014) GAN · Kingma & Welling (2014) VAE · Radford et al. (2016) DCGAN · Ho et al. (2020) DDPM · Arjovsky & Bottou (2017) · Heusel et al. (2017) FID · Xiao et al. (2022) Generative Learning Trilemma · Tishby et al. (2000) Information Bottleneck.
