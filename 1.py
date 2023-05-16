import cv2
import numpy as np
import csv
import fnmatch
import os
import datetime

# Inisialisasi CascadeClassifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Daftar nama orang yang terdaftar
nama_orang = ['DapaArhama']

# Membuka webcam
video_capture = cv2.VideoCapture(0)

# Membuka file CSV untuk mencatat absensi
with open('absensi.csv', mode='a', newline='') as file_csv:
    csv_writer = csv.writer(file_csv)
    csv_writer.writerow(['Nama', 'Waktu Absen'])

    while True:
        # Membaca frame dari webcam
        ret, frame = video_capture.read()

        # Mengubah frame ke dalam skala abu-abu
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mendeteksi wajah pada frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop melalui setiap wajah yang terdeteksi
        for (x, y, w, h) in faces:
            # Gambar kotak di sekitar wajah
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Loop melalui setiap nama orang yang terdaftar
            for i in range(len(nama_orang)):
                # Loop melalui file dalam direktori "data_wajah"
                for file in os.listdir('data_wajah'):
                    # Cocokkan nama file gambar dengan nama orang yang terdaftar
                    if fnmatch.fnmatch(file, f"*{nama_orang[i]}*.jpg"):
                        # Baca gambar
                        img = cv2.imread(os.path.join('data_wajah', file))

                        # Ubah gambar menjadi grayscale
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        # Deteksi wajah pada gambar
                        faces_img = face_cascade.detectMultiScale(gray_img, 1.3, 5)

                        # Loop melalui setiap wajah yang terdeteksi pada gambar
                        for (x_img, y_img, w_img, h_img) in faces_img:
                            # Bandingkan posisi wajah pada frame dengan posisi wajah pada gambar
                            if x > x_img and y > y_img and x + w < x_img + w_img and y + h < y_img + h_img:
                                # Dapatkan waktu absen
                                waktu_absen = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                # Tulis nama dan waktu absen ke file CSV
                                csv_writer.writerow([nama_orang[i], waktu_absen])

                                # Tampilkan nama di sekitar rectangle
                                cv2.putText(frame, nama_orang[i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                break
                        else:
                            # Tampilkan "Tidak Dikenal" jika wajah tidak cocok dengan gambar yang terdaftar
                            cv2.putText(frame, 'Tidak Dikenal', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            break

                            # Tampilkan frame dengan wajah yang terdeteksi
                            cv2.imshow('Webcam', frame)

                      # Jika tombol 'q' ditekan, keluar dari loop
                        cv2.imshow('Deteksi Wajah', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                              break
# Tampilkan gambar dengan wajah yang terdeteksi



video_capture.release()
cv2.destroyAllWindows()
