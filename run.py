import os
import curses
from curses import panel

import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, mean, stddev

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] =  "spark-3.1.2-bin-hadoop3.2"

findspark.init('spark-3.1.2-bin-hadoop3.2')

spark = SparkSession.builder.master('local[*]').getOrCreate()

# Lê todos os csv dentro da pasta dataset
dataset = spark.read.csv('dataset/*', header=True)
working_data = dataset

def get_mean(column):
    """Retorna a média da coluna"""
    return working_data.select(mean(column)).collect()[0][0]

def get_std_deviation(column):
    """Retorna o desvio padrão da coluna"""
    return working_data.select(stddev(column)).collect()[0][0]

def between_dates(begin, end):
    """Retorna um DataFrame com linhas entre as datas, formato: YYYY-mm-dd"""
    return working_data.filter(f"DATE BETWEEN '{begin}' AND '{end}'")

# Desvio padrão coluna temperatura
# print(std_deviation(dataset, 'TEMP'))

# Desvio padrão coluna temperatura entre o ano de 1931 e 1935
# print(std_deviation(between_dates(dataset, '1931-01-01', '1935-12-31'), 'TEMP'))

class MenuConfig:
    def __init__(self, items, stdscreen, title='Title'):
        self.window = stdscreen.subwin(0, 0)
        self.window.keypad(1)
        self.panel = panel.new_panel(self.window)
        self.panel.hide()
        self.title = title
        panel.update_panels()

        self.position = 0
        self.items = items

    def navigate(self, n):
        self.position += n
        if self.position < 0:
            self.position = 0
        elif self.position >= len(self.items):
            self.position = len(self.items) - 1
    
    def display(self):
        self.panel.top()
        self.panel.show()
        self.window.clear()

        while True:
            self.window.refresh()
            curses.doupdate()
            self.window.addstr(0, 1, self.title, curses.A_NORMAL)
            for index, item in enumerate(self.items):
                if index == self.position:
                    mode = curses.A_REVERSE
                else:
                    mode = curses.A_NORMAL

                msg = "%d. %s" % (index, item[0])
                self.window.addstr(2 + index, 1, msg, mode)

            key = self.window.getch()

            if key in [curses.KEY_ENTER, ord("\n")]:
                if self.position == len(self.items) - 1:
                    break
                else:
                    self.items[self.position][1]()

            elif key == curses.KEY_UP:
                self.navigate(-1)

            elif key == curses.KEY_DOWN:
                self.navigate(1)

        self.window.clear()
        self.panel.hide()
        panel.update_panels()
        curses.doupdate()

class Menu:
    def __init__(self, stdscreen):
        self.screen = stdscreen
        curses.curs_set(0)

        menu_range_items = [
            ("Por ano", curses.beep),
            ("Por intervalo de anos", curses.flash),
            ("Voltar", "exit")
        ]
        menu_range = MenuConfig(menu_range_items, self.screen, 'Selecionar subset de dados')

        stddev_show = MenuConfig([("Voltar", "exit")], self.screen, 'Desvio padrão: ')
        
        stddev_menu_items = [
            ("Todas colunas numéricas", curses.beep),
            ("TEMP", lambda: self.settitle_and_display(stddev_show, f'Desvio padrão: {get_std_deviation("TEMP")}')),
            ("DEWP", lambda: self.settitle_and_display(stddev_show, f'Desvio padrão: {get_std_deviation("DEWP")}')),
            ("Voltar", "exit")
        ]
        stddev_menu = MenuConfig(stddev_menu_items, self.screen, 'Dados selecionados: Todos')
        
        main_menu_items = [
            ("Trocar intervalo de dados", menu_range.display),
            ("Desvio Padrão", stddev_menu.display),
            ("Sair", "exit"),
        ]
        main_menu = MenuConfig(main_menu_items, self.screen, 'Dados selecionados: Todos')
        
        main_menu.display()

    def settitle_and_display(self, menu, title):
        menu.title = title
        menu.display()

if __name__ == '__main__':
    curses.wrapper(Menu)
