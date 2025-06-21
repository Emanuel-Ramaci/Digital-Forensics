import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QMessageBox
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from skimage.feature import local_binary_pattern

class ImageForgeryDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Rilevamento Manipolazioni Immagini')
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.load_button = QPushButton('Seleziona Immagine', self)
        self.load_button.clicked.connect(self.seleziona_immagine)
        self.layout.addWidget(self.load_button)

        self.copy_move_button = QPushButton('Rileva Copy-Move', self)
        self.copy_move_button.clicked.connect(self.rileva_copy_move)
        self.layout.addWidget(self.copy_move_button)

        self.inpainting_button = QPushButton('Rileva Inpainting', self)
        self.inpainting_button.clicked.connect(self.rileva_inpainting)
        self.layout.addWidget(self.inpainting_button)

        self.splicing_button = QPushButton('Rileva Splicing', self)
        self.splicing_button.clicked.connect(self.rileva_splicing)
        self.layout.addWidget(self.splicing_button)

        self.setLayout(self.layout)
        self.img = None
        self.img_path = ""

    def seleziona_immagine(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleziona un'immagine", "", "Immagini (*.jpg *.png *.jpeg)")
        if file_path:
            self.img = cv2.imread(file_path)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.img_path = file_path
            QMessageBox.information(self, "Immagine selezionata", "Immagine caricata con successo!")

    def mostra_immagine(self, img, titolo):
        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        plt.title(titolo)
        plt.axis("off")
        plt.show()

    def rileva_copy_move(self):
        if self.img is None:
            QMessageBox.warning(self, "Errore", "Seleziona prima un'immagine!")
            return

        grigio = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY) #Conversione in scala di grigi
        sift = cv2.SIFT_create() #Il SIFT (Scale-Invariant Feature Transform) individua punti chiave distintivi nell'immagine.
                                 #Genera descrittori che rappresentano piccole regioni dell'immagine.
        keypoints, descriptors = sift.detectAndCompute(grigio, None)

        if descriptors is None or len(descriptors) < 2:
            QMessageBox.warning(self, "Errore", "Nessuna caratteristica sufficiente per il rilevamento.")
            return

        clustering = DBSCAN(eps=50, min_samples=2, metric='euclidean').fit(descriptors)
        etichette = clustering.labels_ #DBSCAN (Density-Based Spatial Clustering of Applications with Noise) raggruppa i descrittori simili.
        #Se due regioni hanno descrittori simili, significa che sono copie l'una dell'altra.

        risultato = self.img.copy()
        #Segnalazione delle aree sospette
        for i, label in enumerate(etichette):
            if label != -1:
                x, y = int(keypoints[i].pt[0]), int(keypoints[i].pt[1])
                cv2.circle(risultato, (x, y), 5, (255, 0, 0), -1)
        #Se un punto chiave appartiene a un cluster, viene evidenziato nell'immagine.
        self.mostra_immagine(risultato, "Rilevamento Copy-Move")

    def rileva_inpainting(self):
        if self.img is None:
            QMessageBox.warning(self, "Errore", "Seleziona prima un'immagine!")
            return
        #Conversione in scala di grigi
        grigio = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(grigio, (5, 5), 0) #Il filtro Gaussiano sfoca l'immagine per eliminare il rumore.
                                                   #Se un'area è stata modificata, la sfocatura cambierà in modo anomalo.
        diff = cv2.absdiff(grigio, blur) #Se un'area è stata inpaintata, la differenza sarà elevata.
        #Thresholding per creare una maschera delle anomalie
        _, maschera = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY) #Se la differenza supera una soglia, l'area viene considerata sospetta.

        risultato = self.img.copy()
        risultato[maschera > 0] = [255, 0, 0] #Le aree sospette vengono colorate in rosso.
        
        self.mostra_immagine(risultato, "Rilevamento Inpainting")

    def rileva_splicing(self):
        if self.img is None:
            QMessageBox.warning(self, "Errore", "Seleziona prima un'immagine!")
            return

        # Conversione in spazio di colore YCrCb
        ycrcb = cv2.cvtColor(self.img, cv2.COLOR_RGB2YCrCb)
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2] #I canali Cr (Crominanza Rossa) e Cb (Crominanza Blu) evidenziano variazioni di colore sospette.
        
        # Estrazione delle caratteristiche LBP dai canali Cr e Cb
        lbp_cr = local_binary_pattern(cr, P=8, R=1, method="uniform")
        lbp_cb = local_binary_pattern(cb, P=8, R=1, method="uniform") #LBP (Local Binary Pattern) misura le texture locali, utili per individuare differenze tra parti incollate e originali.
        
        # Calcolo della varianza locale
        varianza_cr = cv2.Laplacian(lbp_cr, cv2.CV_64F).var()
        varianza_cb = cv2.Laplacian(lbp_cb, cv2.CV_64F).var() #Se una parte è stata incollata, la texture e il contrasto cambiano, aumentando la varianza.
        
        # Applicazione del filtro Sobel
        sobelx = cv2.Sobel(cr, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(cr, cv2.CV_64F, 0, 1, ksize=5)
        sobel = np.sqrt(sobelx**2 + sobely**2) #Il filtro Sobel evidenzia i bordi tra zone incollate e originali.
        
        # Normalizzazione del contrasto
        sobel_norm = (sobel / sobel.max() * 255).astype(np.uint8)
        contrast = cv2.convertScaleAbs(sobel_norm, alpha=1.5, beta=0)  #Aumenta la visibilità delle aree sospette.
        
        # Thresholding adattivo basato sulla varianza
        soglia = max(30, min(120, int((varianza_cr + varianza_cb) / 40))) 
        _, maschera = cv2.threshold(contrast, soglia, 255, cv2.THRESH_BINARY) #Se la varianza è elevata, la soglia viene abbassata per aumentare la sensibilità.
        
        # Segmentazione delle aree sospette
        contorni, _ = cv2.findContours(maschera, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        risultato = self.img.copy()
        
        # Filtra contorni troppo piccoli
        contorni_filtrati = [cnt for cnt in contorni if cv2.contourArea(cnt) > 600]  # I contorni troppo piccoli vengono scartati per evitare falsi positivi.

        #Disegno dei contorni sulle zone sospette
        cv2.drawContours(risultato, contorni_filtrati, -1, (255, 0, 0), 2) # Le aree incollate vengono evidenziate nell'immagine
        
        self.mostra_immagine(risultato, "Rilevamento Splicing")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageForgeryDetector()
    window.show()
    sys.exit(app.exec_())
