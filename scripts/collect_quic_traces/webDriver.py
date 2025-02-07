from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def open_website(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--user-data-dir=/users/KevinHZ/traces/google/")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    print(driver.title)

open_website("https://www.google.com/")
