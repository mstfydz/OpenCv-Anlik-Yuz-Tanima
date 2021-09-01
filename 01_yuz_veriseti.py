# Original code from https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826

import cv2

kamera = cv2.VideoCapture(0)
kamera.set(3, 640) # video genişliğini belirle
kamera.set(4, 480) # video yüksekliğini belirle
face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
# Her farklı kişi için farklı bir yüz tamsayısı ata
# face_id = input('\n enter user id end press <return> ==>  ')
MAXFOTOSAY = 50 # Her bir yüz için kullanılacak imaj sayısı
face_id = 1
print("\n [INFO] Kayıtlar başlıyor. Kameraya bak ve bekle ...")

say = 0

while(True):
    ret, img = kamera.read()
    # img = cv2.flip(img, -1) # gerekiyorsa kullan
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yuzler = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in yuzler:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        say += 1
        # Yakalanan imajı veriseti klasörüne kaydet
        cv2.imwrite("veriseti/" + str(face_id) + '.' + str(say) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('imaj', img)
        print("Kayıt no: ",say)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif say >= MAXFOTOSAY:
         break
# Belleği temizle
print("\n [INFO] Program sonlanıyor ve bellek temizleniyor.")
kamera.release()
cv2.destroyAllWindows()