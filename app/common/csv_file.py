import codecs
import csv


class CSVFile:

    @staticmethod
    def load(file_path: str) -> list:
        """
        CSVファイルを読み込む
        """
        rows = []
        with codecs.open(file_path, "r", "utf8") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
        return rows

    @staticmethod
    def save(file_path: str, rows: list):
        """
        CSVファイルに書き込む
        """
        with codecs.open(file_path, "w", "utf8") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
