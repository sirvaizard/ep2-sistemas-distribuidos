# Download all datasets
from bs4 import BeautifulSoup
import requests
from pathlib import Path
import threading

s = requests.Session()

base_url = 'https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/'

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 10, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(end=f'\r{prefix} |{bar}| {percent}% {suffix}')
    # Print New Line on Complete
    if iteration >= total: 
        print()
total_files = 0
downloaded_files = 0

def download_csv(url, filename, year):
  global downloaded_files
  global total_files
  printProgressBar(downloaded_files, total_files, suffix=f'file {downloaded_files} of {total_files}')
  csv = s.get(url)
  with open(f'./dataset/{year}/{filename}', 'w') as f:
    f.write(csv.text)
  downloaded_files += 1

threads = []

total = 1937 - 1929
print('getting all years files links')
for idx, year in enumerate(range(1929, 1938)):
  printProgressBar(idx, total, suffix=f'current year: {year}')
  url_with_year = base_url + str(year) + "/"
  raw_html = s.get(url_with_year).text
  soup = BeautifulSoup(raw_html, 'html.parser')
  trs = soup.select('tr')[3:]

  Path(f'./dataset/{year}/').mkdir(parents=True, exist_ok=True)
  for tr in trs:
    if tr.a:
      t = threading.Thread(target=download_csv, args=(url_with_year + tr.a.text, tr.a.text, year))
      t.deamon = True
      threads.append(t)
      total_files += 1

print('\ndownloading csv files')
for t in threads:
  t.start()

for t in threads:
  t.join()