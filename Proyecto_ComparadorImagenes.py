import unittest
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity

class Automatizacion(unittest.TestCase):
    def setUp(self):
        self.driver= webdriver.Edge(EdgeChromiumDriverManager().install())
    
    def test_ComparandolaFoto(self):
        driver= self.driver
        driver.get("https://es.wikipedia.org/wiki/Búsqueda_de_errores")
        time.sleep(3)
        #driver.execute_script("window.scrollTo(0, 1200)") #Para hacer scroll
        #time.sleep(3)
        Imagen= driver.find_element_by_xpath('/html/body/div[3]/div[3]/div[5]/div[1]/div[2]/div/a/img')
        time.sleep(3)
        Imagen.click()
        time.sleep(3)
        Imagen= driver.find_element_by_xpath('/html/body/div[7]/div/div[2]/div/div[1]/img')
        Imagen.screenshot("Selenium/imgA.png")
        #BuscandoImagen= driver.find_element_by_xpath('//*[@id="post-55149"]/div/p[10]/a[1]/img')
        #BuscandoImagen.screenshot("Selenium/imgC.png")
        #BuscandoImagen2= driver.find_element_by_xpath('//*[@id="post-55149"]/div/p[10]/a[2]/img')
        #BuscandoImagen2.screenshot("Selenium/imgC2.png")
        
        imgA= cv2.imread("Selenium/imgA.png")
        h, w, channels= imgA.shape
        half= w//2
        parteA= imgA[:, :half]
        parteB= imgA[:, half:]
        cv2.imwrite("Selenium/ImagenA.png", parteA)
        cv2.imwrite("Selenium/ImagenB.png", parteB)
        imgC = cv2.imread('Selenium/ImagenA.png')
        imgC2 = cv2.imread('Selenium/ImagenB.png')

        parteA_gray = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
        parteB_gray = cv2.cvtColor(imgC2, cv2.COLOR_BGR2GRAY)

        (puntuacion, diff) = structural_similarity(parteA_gray, parteB_gray, full=True)
        print("Similitud de Imagen", puntuacion)

        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(imgC.shape, dtype='uint8')
        filled_after = imgC2.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area > 20:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(imgC, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.rectangle(imgC2, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.drawContours(mask, [c], 0, (0,255,0), -1)
                cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

        cv2.imshow('Original', imgA)
        cv2.imshow('Imagen', imgC)
        cv2.imshow('Imagen2', imgC2)
        cv2.imshow('diff',diff)
        cv2.imshow('mask',mask)
        cv2.imshow('rellenado posterior',filled_after)
        cv2.waitKey(0)
    
    def tearDown(self):
        self.driver.quit()

if __name__=="__main__":
    unittest.main()