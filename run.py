import os
import curses
from curses import panel

import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, mean, stddev, year, month
import matplotlib.pyplot as plt

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "spark-3.1.2-bin-hadoop3.2"

findspark.init('spark-3.1.2-bin-hadoop3.2')

spark = SparkSession.builder.master('local[*]').getOrCreate()

# Lê todos os csv dentro da pasta dataset
dataset = spark.read.csv('dataset/*', header=True)
working_data = dataset

def getShowString(df, truncate=True, vertical=False):
    if isinstance(truncate, bool) and truncate:
        return(df._jdf.showString(df.count(), 0, vertical))
    else:
        return(df._jdf.showString(df.count(), int(truncate), vertical))

def data_describe(df, column, group):

  from pyspark.sql.functions import mean, stddev, count, max, min, kurtosis, skewness, year, month

  if group == "all":
    result = df.agg(count(column), mean(column), min(column), max(column), skewness(column), kurtosis(column), stddev(column))
    return getShowString(result)

  if group == "year":
    result = df.groupBy(year('DATE')).agg(count(column), mean(column), min(column), max(column), skewness(column), kurtosis(column), stddev(column)).orderBy(year('DATE'))
    return getShowString(result)

  if group == "month":
    result = df.groupBy([year('DATE') , month('DATE')]).agg(count(column), mean(column), min(column), max(column), skewness(column), kurtosis(column), stddev(column)).orderBy([year('DATE'), month('DATE')])
    return getShowString(result)


def get_mean(column):
    """Retorna a média da coluna"""
    return working_data.select(mean(column)).collect()[0][0]


def get_std_deviation(column):
    """Retorna o desvio padrão da coluna"""
    return working_data.select(stddev(column)).collect()[0][0]


def between_dates(begin, end):
    """Retorna um DataFrame com linhas entre as datas, formato: YYYY-mm-dd"""
    return working_data.filter(f"DATE BETWEEN '{begin}' AND '{end}'")


def set_interval(date):
    global working_data
    begin, end = date.split('-')
    
    begin_splitted = begin.split('/')
    end_splitted = end.split('/')
        
    if len(begin_splitted) > 1:
        begin_parsed = f'{begin_splitted[2]}-{begin_splitted[1]}-{begin_splitted[0]}'
        end_parsed = f'{end_splitted[2]}-{end_splitted[1]}-{end_splitted[0]}'
    else:
        begin_parsed = f'{begin}-01-01'
        end_parsed = f'{end}-12-31'
    working_data = dataset.filter(f"DATE BETWEEN '{begin_parsed}' AND '{end_parsed}'")

def restore_interval():
    global working_data
    working_data = dataset
    MenuConfig.interval = 'Todos'

def plot_graph(df, column, group, x_label, y_label):
    if group == 'year':
        data = df.groupBy(year('DATE')).agg(mean(column)).orderBy(year('DATE'))
    if group == 'month':
        data = df.groupBy(month('DATE')).agg(mean(column)).orderBy(month('DATE'))

    x = data.select(f'{group}(DATE)').rdd.flatMap(lambda x: x).collect()
    y = data.select(f'avg({column})').rdd.flatMap(lambda x: x).collect()

    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

class MenuConfig:
    interval = 'Todos'

    def __init__(self, items, stdscreen):
        self.window = stdscreen.subwin(0, 0)
        self.window.keypad(1)
        self.window.scrollok(True)
        self.panel = panel.new_panel(self.window)
        self.panel.hide()
        self.page = 0
        self.data = ''
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
        self.page = 3

        while True:
            try:
                rows, cols = self.window.getmaxyx()

                self.window.clrtobot()
                self.window.refresh()
                curses.doupdate()
                self.window.addstr(0, 1, f'Dados selecionados: {MenuConfig.interval}', curses.A_NORMAL)
                start = 3
                for line in self.data.split('\n')[self.page:self.page+(rows - 5)]:
                    self.window.addstr(start, 1, line, curses.A_NORMAL)
                    start += 1
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
                elif key == curses.KEY_NPAGE:
                    self.page += 2
                elif key == curses.KEY_PPAGE:
                    self.page -= 2
                    if self.page < 0:
                        self.page = 0
            except:
                pass

        self.window.clear()
        self.panel.hide()
        panel.update_panels()
        curses.doupdate()


class Menu:
    def __init__(self, stdscreen):
        self.screen = stdscreen
        self.column = ''
        curses.curs_set(0)

        menu_range_items = [
            ("Todos", restore_interval),
            ("Por ano, ex: 1929-1930", self.get_interval),
            ("Por data completa, ex: 01/01/1929-03/03/1930", self.get_interval),
            ("Voltar", "exit")
        ]
        menu_range = MenuConfig(
            menu_range_items, self.screen)

        ## describe
        scroll_txt = ' (Page Down e Page Up para mover pra cima e para baixo)'
        describe_menu_groupy = MenuConfig([
            ('Todos',  lambda: self.settitle_and_display(
                describe_show, f'{self.column} - Todos os dados' + scroll_txt, f'\n\n\n{data_describe(working_data, self.column, "all")}')),
            ('Por ano', lambda: self.settitle_and_display(
                describe_show, f'{self.column} - Agrupado por ano' + scroll_txt, f'\n\n\n{data_describe(working_data, self.column, "year")}')),
            ('Por mês', lambda: self.settitle_and_display(
                describe_show, f'{self.column} - Agrupado por mês' + scroll_txt, f'\n\n\n{data_describe(working_data, self.column, "month")}')),
            ('Voltar', 'exit')
        ], self.screen)


        describe_show = MenuConfig([("Voltar", "exit")], self.screen)

        describe_menu_items = [
            ("TEMP", lambda: self.set_describe_menu('TEMP', describe_menu_groupy)),
            ("DEWP", lambda: self.set_describe_menu('DEWP', describe_menu_groupy)),
            ("SLP", lambda: self.set_describe_menu('SLP', describe_menu_groupy)),
            ("STP", lambda: self.set_describe_menu('STP', describe_menu_groupy)),
            ("VISIB", lambda: self.set_describe_menu('VISIB', describe_menu_groupy)),
            ("WDSP", lambda: self.set_describe_menu('WDSP', describe_menu_groupy)),
            ("MXSPD", lambda: self.set_describe_menu('MXSPD', describe_menu_groupy)),
            ("GUST", lambda: self.set_describe_menu('GUST', describe_menu_groupy)),
            ("MAX", lambda: self.set_describe_menu('MAX', describe_menu_groupy)),
            ("MIN", lambda: self.set_describe_menu('MIN', describe_menu_groupy)),
            ("PRCP", lambda: self.set_describe_menu('PRCP', describe_menu_groupy)),
            ("SNDP", lambda: self.set_describe_menu('SNDP', describe_menu_groupy)),
            ('Voltar', 'exit')
        ]

        describe_menu = MenuConfig(describe_menu_items, self.screen)

        # Plotar graficos
        plot_menu_groupy = MenuConfig([
            ('Agrupado por ano',  lambda: plot_graph(working_data, self.column, 'year', 'Ano', self.column)),
            ('Agrupado por mês', lambda: plot_graph(working_data, self.column, 'month', 'Mês', self.column)),
            ('Voltar', 'exit')
        ], self.screen)

        plot_menu_items = [
            ("TEMP", lambda: self.set_describe_menu('TEMP', plot_menu_groupy)),
            ("DEWP", lambda: self.set_describe_menu('DEWP', plot_menu_groupy)),
            ("SLP", lambda: self.set_describe_menu('SLP', plot_menu_groupy)),
            ("STP", lambda: self.set_describe_menu('STP', plot_menu_groupy)),
            ("VISIB", lambda: self.set_describe_menu('VISIB', plot_menu_groupy)),
            ("WDSP", lambda: self.set_describe_menu('WDSP', plot_menu_groupy)),
            ("MXSPD", lambda: self.set_describe_menu('MXSPD', plot_menu_groupy)),
            ("GUST", lambda: self.set_describe_menu('GUST', plot_menu_groupy)),
            ("MAX", lambda: self.set_describe_menu('MAX', plot_menu_groupy)),
            ("MIN", lambda: self.set_describe_menu('MIN', plot_menu_groupy)),
            ("PRCP", lambda: self.set_describe_menu('PRCP', plot_menu_groupy)),
            ("SNDP", lambda: self.set_describe_menu('SNDP', plot_menu_groupy)),
            ('Voltar', 'exit')
        ]
        plot_menu = MenuConfig(plot_menu_items, self.screen)
        #

        ##
        main_menu_items = [
            ("Trocar intervalo de dados", menu_range.display),
            ("Describe", describe_menu.display),
            ('Plotar gráficos', plot_menu.display),
            ("Sair", "exit")
        ]
        main_menu = MenuConfig(main_menu_items, self.screen)

        main_menu.display()

    def settitle_and_display(self, menu, title, data=''):
        menu.title = title
        menu.data = data
        menu.display()

    def set_describe_menu(self, column, menu):
        self.column = column
        menu.display()

    def get_interval(self):
        curses.echo()
        self.screen.addstr(5, 0, "Digite a data: ")
        self.screen.refresh()
        date = self.screen.getstr(6,0, 21)
        set_interval(date.decode())
        self.screen.refresh()
        MenuConfig.interval = date.decode()


if __name__ == '__main__':
    curses.wrapper(Menu)