from django.db import connection


class classified:
    def all(self):
        with connection.cursor() as cursor:
            cursor.execute("select image_path from classified")
            datas= cursor.fetchall()
            print(datas)
        return datas

    # def upload(self):
    #     with connection.cursor() as cursor:
    #         sql="""INSERT INTO classified