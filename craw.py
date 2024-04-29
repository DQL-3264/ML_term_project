import requests
import urllib
from bs4 import BeautifulSoup
from time import sleep
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
Edge_op = Options()
# Edge_op.add_experimental_option('useAutomationExtension', False)
# \隱藏「正由自動化程序控制」這行字
Edge_op.add_experimental_option('excludeSwitches', ['enable-automation'])
driver = webdriver.Edge(options=Edge_op)
driver_wait = WebDriverWait(driver, 20, 1)
driver.get('https://www.istockphoto.com/hk/search/2/image-film?phrase=%E4%BA%9E%E6%B4%B2%E4%BA%BA&page=15')
soup = BeautifulSoup(driver.page_source, "html.parser")
count = 946
for i in driver.find_elements(By.TAG_NAME, "img"):
        item_img = i.get_attribute("src")
        # print(item_img)
        with open("./image/"+str(count)+".jpg", mode="wb")as f:
            f.write(requests.get(item_img).content)
        count = count+1
print(len(driver.find_elements(By.TAG_NAME, "img")))
for i in driver.find_elements(By.TAG_NAME, "img"):
    item_img = i.get_attribute("src")
    with open("./image/"+str(count)+".jpg", mode="wb")as f:
        f.write(requests.get(item_img).content)
    count = count+1
    # 抓到15面
    print(driver.find_elements(By.TAG_NAME, "img")[5].get_attribute("src"))
    button = driver.find_element(By.XPATH, '/html/body/div[2]/section/div/main/div/div/div[2]/div/section/button[2]')
    button.click()
    sleep(20)
    driver_wait.until(EC.presence_of_element_located((By.TAG_NAME, "img")))
for j in range(10):
# print(len(driver.find_elements(By.CLASS_NAME, '_aagv')))
    for i in driver.find_elements(By.TAG_NAME, "img"):
        
        print(i.find_element(By.CLASS_NAME, '_aagv'))
        item_img = i.get_attribute("alt")
        print(item_img)
        with open("./image/"+str(count)+".jpg", mode="wb")as f:
            f.write(requests.get(item_img).content)
        count = count+1
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        print(driver.find_elements(By.TAG_NAME, "img")[5].get_attribute("src"))
        button = driver.find_element(By.XPATH, '/html/body/div[2]/section/div/main/div/div/div[4]/div/section/button[2]')
        button.click()
        sleep(20)
        driver_wait.until(EC.presence_of_element_located((By.TAG_NAME, "img")))
