import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Chargement et normalisation de l'image dans [0, 1]
def load_image(path, gray=True):
    img = Image.open(path)
    if gray:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    return np.array(img, dtype=np.float64) / 255.0

def fuzzy_c_means(X, C=2, m=2.0, eps=0.01, max_iter=100):
    N, d = X.shape
    
    # Initialisation aléatoire de U (chaque ligne somme à 1)
    U = np.random.rand(N, C)
    U = U / np.sum(U, axis=1, keepdims=True)
    
    for k in range(max_iter):
        U_prev = U.copy()
        
        # Calcul des centres c_j
        um = U ** m
        centers = np.zeros((C, d))
        for j in range(C):
            num = np.sum(um[:, j:j+1] * X, axis=0)
            den = np.sum(um[:, j])
            centers[j] = num / den
            
        # Mise à jour de U
        dist = np.zeros((N, C))
        for j in range(C):
            dist[:, j] = np.linalg.norm(X - centers[j], axis=1)
            
        # Evite la division par zéro ou sous-dépassement numérique
        dist = np.fmax(dist, 1e-10)
        
        puissance = 2.0 / (m - 1.0)
        for j in range(C):
            denom = np.zeros(N)
            for c_k in range(C):
                denom += (dist[:, j] / dist[:, c_k]) ** puissance
            U[:, j] = 1.0 / denom
            
        # Critère d'arrêt : max |U^(k+1) - U^(k)| < epsilon
        diff = np.max(np.abs(U - U_prev))
        print(f"Iteration {k+1} - erreur max: {diff:.5f}")
        if diff < eps:
            print(f"Convergence atteinte à l'itération {k+1}")
            break
            
    # Tri des clusters du plus sombre au plus clair
    order = np.argsort(np.mean(centers, axis=1))
    centers = centers[order]
    U = U[:, order]

    return centers, U

if __name__ == "__main__":
    np.random.seed(42)
    
    # Image en niveaux de gris (K = 2)
    print("--- Test sur l'image en niveaux de gris (C=2) ---")
    img_gray = load_image("milky-way-nvg.jpg", gray=True)
    H, W = img_gray.shape
    
    X_gray = img_gray.reshape(-1, 1)
    
    C = 2
    centers_gray, U_gray = fuzzy_c_means(X_gray, C=C, m=2.0, eps=0.01)
    print("Centres gris :", np.round(centers_gray.flatten(), 4))
    
    U_gray_img = U_gray.reshape(H, W, C)
    
    # Segmentation dure par argmax et reconstruction par intensité des centres
    labels_gray = np.argmax(U_gray_img, axis=2)
    seg_gray = centers_gray[labels_gray].reshape(H, W)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Image originale")
    plt.axis('off')
    
    for j in range(C):
        plt.subplot(1, 4, j + 2)
        plt.imshow(U_gray_img[:, :, j], cmap='jet', vmin=0, vmax=1)
        plt.title(f"Heatmap Cluster {j+1}")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
    plt.subplot(1, 4, 4)
    plt.imshow(seg_gray, cmap='gray')
    plt.title("Segmentation (argmax)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("resultat_nvg_c2.png", dpi=150)
    plt.show()
    plt.close()

    # Image couleur RGB (K = 3)
    print("\n--- Test sur l'image couleur (C=3) ---")
    img_rgb = load_image("milky-way.jpg", gray=False)
    H_rgb, W_rgb, _ = img_rgb.shape
    
    X_rgb = img_rgb.reshape(-1, 3)
    
    C_rgb = 3
    centers_rgb, U_rgb = fuzzy_c_means(X_rgb, C=C_rgb, m=2.0, eps=0.01)
    print("Centres RGB :\n", np.round(centers_rgb, 4))
    
    U_rgb_img = U_rgb.reshape(H_rgb, W_rgb, C_rgb)
    labels_rgb = np.argmax(U_rgb_img, axis=2)
    
    # Reconstruction avec la couleur de chaque centre
    seg_rgb = centers_rgb[labels_rgb]
    
    plt.figure(figsize=(14, 4))
    plt.subplot(1, C_rgb + 2, 1)
    plt.imshow(img_rgb)
    plt.title("Originale RGB")
    plt.axis('off')
    
    for j in range(C_rgb):
        plt.subplot(1, C_rgb + 2, j + 2)
        plt.imshow(U_rgb_img[:, :, j], cmap='jet', vmin=0, vmax=1)
        plt.title(f"Heatmap C{j+1}")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
    plt.subplot(1, C_rgb + 2, C_rgb + 2)
    plt.imshow(seg_rgb)
    plt.title("Segmentation")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("resultat_rgb_c3.png", dpi=150)
    plt.show()
    plt.close()
