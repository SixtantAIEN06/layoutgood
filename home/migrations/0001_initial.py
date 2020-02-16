# Generated by Django 2.2.10 on 2020-02-16 08:51

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Classified',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('person', models.IntegerField()),
                ('bicycle', models.IntegerField()),
                ('car', models.IntegerField()),
                ('motorbike', models.IntegerField()),
                ('aeroplane', models.IntegerField()),
                ('bus', models.IntegerField()),
                ('train', models.IntegerField()),
                ('truck', models.IntegerField()),
                ('boat', models.IntegerField()),
                ('traffic_light', models.IntegerField()),
                ('fire_hydrant', models.IntegerField()),
                ('stop_sign', models.IntegerField()),
                ('parking_meter', models.IntegerField()),
                ('bench', models.IntegerField()),
                ('bird', models.IntegerField()),
                ('cat', models.IntegerField()),
                ('dog', models.IntegerField()),
                ('horse', models.IntegerField()),
                ('sheep', models.IntegerField()),
                ('cow', models.IntegerField()),
                ('elephant', models.IntegerField()),
                ('bear', models.IntegerField()),
                ('zebra', models.IntegerField()),
                ('giraffe', models.IntegerField()),
                ('backpack', models.IntegerField()),
                ('umbrella', models.IntegerField()),
                ('handbag', models.IntegerField()),
                ('tie', models.IntegerField()),
                ('suitcase', models.IntegerField()),
                ('frisbee', models.IntegerField()),
                ('skis', models.IntegerField()),
                ('snowboard', models.IntegerField()),
                ('sports_ball', models.IntegerField()),
                ('kite', models.IntegerField()),
                ('baseball_bat', models.IntegerField()),
                ('baseball_glove', models.IntegerField()),
                ('skateboard', models.IntegerField()),
                ('surfboard', models.IntegerField()),
                ('tennis_racket', models.IntegerField()),
                ('bottle', models.IntegerField()),
                ('wine_glass', models.IntegerField()),
                ('cup', models.IntegerField()),
                ('fork', models.IntegerField()),
                ('knife', models.IntegerField()),
                ('spoon', models.IntegerField()),
                ('bowl', models.IntegerField()),
                ('banana', models.IntegerField()),
                ('apple', models.IntegerField()),
                ('sandwich', models.IntegerField()),
                ('orange', models.IntegerField()),
                ('broccoli', models.IntegerField()),
                ('carrot', models.IntegerField()),
                ('hot_dog', models.IntegerField()),
                ('pizza', models.IntegerField()),
                ('donut', models.IntegerField()),
                ('cake', models.IntegerField()),
                ('chair', models.IntegerField()),
                ('sofa', models.IntegerField()),
                ('pottedplant', models.IntegerField()),
                ('bed', models.IntegerField()),
                ('diningtable', models.IntegerField()),
                ('toilet', models.IntegerField()),
                ('tvmonitor', models.IntegerField()),
                ('laptop', models.IntegerField()),
                ('mouse', models.IntegerField()),
                ('remote', models.IntegerField()),
                ('keyboard', models.IntegerField()),
                ('cell_phone', models.IntegerField()),
                ('microwave', models.IntegerField()),
                ('oven', models.IntegerField()),
                ('toaster', models.IntegerField()),
                ('sink', models.IntegerField()),
                ('refrigerator', models.IntegerField()),
                ('book', models.IntegerField()),
                ('clock', models.IntegerField()),
                ('vase', models.IntegerField()),
                ('scissors', models.IntegerField()),
                ('teddy_bear', models.IntegerField()),
                ('hair_drier', models.IntegerField()),
                ('toothbrush', models.IntegerField()),
                ('potted_plant', models.IntegerField()),
                ('motorcycle', models.IntegerField()),
                ('airplane', models.IntegerField()),
                ('dining_table', models.IntegerField()),
                ('landscape', models.IntegerField()),
                ('couch', models.IntegerField()),
                ('tv', models.IntegerField()),
                ('image_path', models.TextField()),
            ],
            options={
                'db_table': 'classified',
                'managed': False,
            },
        ),
    ]