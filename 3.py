import cv2
import csv
import datetime

# Inisialisasi detektor wajah menggunakan Haar cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Daftar nama orang yang terdaftar
nama_orang = ['DapaArhama']

# Menginisialisasi variabel absensi
absensi = {}

# Membuka webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Membaca frame dari webcam
    ret, frame = video_capture.read()

    # Mengubah frame ke dalam skala abu-abu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah pada frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop melalui setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Gambar kotak dan tulis nama di sekitar wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Tampilkan frame dengan wajah yang terdeteksi
    cv2.imshow('Webcam', frame)

    # Jika tombol 'q' ditekan, keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Simpan data absensi ke file CSV
with open('absensi.csv', mode='w', newline='') as file_csv:
    csv_writer = csv.writer(file_csv)
    csv_writer.writerow(['Nama', 'Waktu Absen'])
    for nama, waktu_absen in absensi.items():
        csv_writer.writerow([nama, waktu_absen])

# Tutup webcam dan tutup window
video_capture.release()
cv2.destroyAllWindows()
