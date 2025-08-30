import numpy as np
import matplotlib.pyplot as plt

# muat dataset
dataset_path = r"C:\Users\ferda\Tugas-Akhir\dataset\chestmnist.npz"
data = np.load(dataset_path)

# ambil label training
train_labels = data['train_labels']
total_sampel = len(train_labels)

# nama penyakit (label asli)
nama_penyakit = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

print("Analisis Imbalance Dataset ChestMNIST")
print("=" * 45)
print(f"Total sampel training: {total_sampel:,} gambar\n")

# hitung distribusi dan rasio imbalance
jumlah_positif = []
jumlah_negatif = []
rasio_imbalance = []

# hitung kasus normal (tidak ada label positif)
labels_per_gambar = np.sum(train_labels, axis=1)
kasus_normal = np.sum(labels_per_gambar == 0)
kasus_abnormal = total_sampel - kasus_normal

print("Distribusi Normal vs Abnormal:")
print("-" * 35)
print(f"Normal (0 label)     : {kasus_normal:6,} ({(kasus_normal/total_sampel)*100:5.1f}%)")
print(f"Abnormal (>=1 label) : {kasus_abnormal:6,} ({(kasus_abnormal/total_sampel)*100:5.1f}%)")
print(f"Rasio Normal:Abnormal: 1:{kasus_abnormal/kasus_normal:.1f}")
print()

print("Distribusi Label (Train Set):")
print("-" * 45)

for i, penyakit in enumerate(nama_penyakit):
    positif = int(np.sum(train_labels[:, i]))
    negatif = total_sampel - positif
    persentase_positif = (positif / total_sampel) * 100
    rasio = negatif / positif if positif > 0 else float('inf')
    
    jumlah_positif.append(positif)
    jumlah_negatif.append(negatif)
    rasio_imbalance.append(rasio)
    
    print(f"{penyakit:18}: {positif:6,} ({persentase_positif:5.1f}%) | Rasio: 1:{rasio:.1f}")

print("\nTingkat Imbalance:")
print("-" * 25)

# kategorisasi tingkat imbalance
for i, (penyakit, rasio) in enumerate(zip(nama_penyakit, rasio_imbalance)):
    if rasio < 5:
        kategori = "Seimbang"
    elif rasio < 10:
        kategori = "Ringan"  
    elif rasio < 50:
        kategori = "Sedang"
    else:
        kategori = "Berat"
    
    print(f"{penyakit:18}: {kategori:8} (1:{rasio:.1f})")

# statistik imbalance
print(f"\nStatistik Imbalance:")
print(f"Rasio terendah  : 1:{min(rasio_imbalance):.1f} ({nama_penyakit[rasio_imbalance.index(min(rasio_imbalance))]})")
print(f"Rasio tertinggi : 1:{max(rasio_imbalance):.1f} ({nama_penyakit[rasio_imbalance.index(max(rasio_imbalance))]})")
print(f"Rata-rata rasio : 1:{np.mean(rasio_imbalance):.1f}")

# visualisasi distribusi
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# bar chart jumlah sampel positif
bars1 = ax1.bar(range(len(nama_penyakit)), jumlah_positif, color='steelblue')
ax1.set_xlabel('Penyakit')
ax1.set_ylabel('Jumlah Kasus Positif')
ax1.set_title('Distribusi Jumlah Kasus Positif')
ax1.set_xticks(range(len(nama_penyakit)))
ax1.set_xticklabels(nama_penyakit, rotation=45, ha='right')

# tambahkan angka di atas bar
for bar, nilai in zip(bars1, jumlah_positif):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             f'{nilai:,}', ha='center', va='bottom', fontsize=8)

# bar chart rasio imbalance  
bars2 = ax2.bar(range(len(nama_penyakit)), rasio_imbalance, color='orangered')
ax2.set_xlabel('Penyakit')
ax2.set_ylabel('Rasio Imbalance (Negatif:Positif)')
ax2.set_title('Tingkat Imbalance per Penyakit')
ax2.set_xticks(range(len(nama_penyakit)))
ax2.set_xticklabels(nama_penyakit, rotation=45, ha='right')
ax2.set_yscale('log')

# tambahkan garis threshold
ax2.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Imbalance Sedang')
ax2.axhline(y=50, color='darkred', linestyle='--', alpha=0.7, label='Imbalance Berat')
ax2.legend()

plt.tight_layout()
chart_path = r"C:\Users\ferda\Tugas-Akhir\dataset\distribusi_imbalance.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\nGrafik distribusi disimpan di: {chart_path}")

data.close()