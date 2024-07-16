import pandas as pd

# Creating a class to read any type of file using pandas
class DataLoader:

    # constructor class
    def __init__(self,path):
        self.path=path
        
    # function to handle csv files
    def csv_file(self,sep,encode):
        try:
            data = pd.read_csv(self.path,sep=sep,encoding=encode)
            return data
        except Exception as e:
            raise ValueError(f"While reading the csv file we are getting Error as : {e}")

    # function to handle Excel files
    def excel_file(self,sheetnum):
        try:
            data = pd.read_excel(self.path,sheet_name=sheetnum)
            print(f"Data is from the sheet {self.sheet}, if you want other sheet data pass the sheet number along with the file-path!")
            return data
        except Exception as x:
            raise ValueError(f"While Reading the file there was an error: {x}")
            

    # function to handle html files
    def html_file(self):
        try:
            data  = pd.read_html(self.path)
            return data
        except Exception as p:
            raise ValueError(f"While reading the html file getting Error as: {p}")


    # funtion to handle json files
    def json_file(self):
        try:
            data = pd.read_json(self.path)
            return data
        except Exception as c:
            raise ValueError(f"While reading the Json getting error as: {c}")
            

    # funtion the to any type of files
    def read_data(self,sheet_name=0,sep=None,encoding = 'utf-8'):
        self.sheet_name = sheet_name
        self.sep = sep
        self.encoding = encoding
        try:
            if self.path.endswith(".csv") or self.path.endswith(".txt"):
                data_frame = self.csv_file(self.sep,self.encoding)
                return data_frame
            elif self.path.endswith(".html"):
                data_frame = self.read_html(self.sheetname)
                return data_frame
            elif self.path.endswith(".xlsx"):
                data_frame = self.excel_file()
                return data_frame
            elif self.path.endswith(".json"):
                dataframe = self.json_file()
                return data_frame
            else:
                raise ValueError("The File is not in .csv,.html,.json,.xlsx Extensions")
        except Exception as e:
            print(f"Error readin the file: {e}")