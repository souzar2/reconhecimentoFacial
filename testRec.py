import cv2
import dlib
import cmake
import face_recognition as fr

from PIL import Image

def main():
    print("Se preferir, é possível digitar o caminho completo da imagem que será utilizada como teste...")
    print()
    print("Se não, a foto utilizada como padrão do teste será a do Tony Stark")

    caminho_arquivo = input("Digite o caminho completo da imagem: ")

    try:
        imgTest = fr.load_image_file(caminho_arquivo)
        # Faça o processamento necessário com a imagem aqui
        print("Imagem carregada com sucesso!")
    except Exception as e:
        imgTest = fr.load_image_file('Tony.jpg')
        print("Erro ao abrir a imagem, a do Tony Stark será utilizada para o teste")

    video_cap = cv2.VideoCapture(0)
    rostoCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cont = 0
    imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

    corBorda = [255, 0, 0]

    while True:
        ret, frame = video_cap.read()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rostos = rostoCascade.detectMultiScale(
            rgb,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if(cont%40==0):
            comparacao = False
                
            rostoEncondings = fr.face_encodings(rgb)
            encodeTest = fr.face_encodings(imgTest)
            
            if(len(rostoEncondings)>0 and len(encodeTest)>0):
                comparacao = fr.compare_faces([rostoEncondings[0]], encodeTest[0])[0]

        if comparacao:
            corBorda = [0, 255, 0]
        else:
            corBorda = [0, 0, 255]
        
        for (x, y, w, h) in rostos:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (corBorda), 2)

        cv2.imshow('video', frame)

        cont += 1


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video_cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
