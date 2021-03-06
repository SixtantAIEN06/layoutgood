# Generated by Django 2.2.10 on 2020-02-22 08:02

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Members',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=255)),
                ('user_id', models.CharField(max_length=255, unique=True)),
                ('social_id', models.CharField(max_length=255)),
                ('password', models.CharField(max_length=255)),
                ('phone_number', models.CharField(max_length=255)),
                ('email', models.EmailField(max_length=254)),
            ],
            options={
                'db_table': 'members',
                'managed': True,
            },
        ),
        migrations.CreateModel(
            name='Classified',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('person', models.IntegerField(default=0)),
                ('bicycle', models.IntegerField(default=0)),
                ('car', models.IntegerField(default=0)),
                ('motorbike', models.IntegerField(default=0)),
                ('aeroplane', models.IntegerField(default=0)),
                ('bus', models.IntegerField(default=0)),
                ('train', models.IntegerField(default=0)),
                ('truck', models.IntegerField(default=0)),
                ('boat', models.IntegerField(default=0)),
                ('traffic_light', models.IntegerField(default=0)),
                ('fire_hydrant', models.IntegerField(default=0)),
                ('stop_sign', models.IntegerField(default=0)),
                ('parking_meter', models.IntegerField(default=0)),
                ('bench', models.IntegerField(default=0)),
                ('bird', models.IntegerField(default=0)),
                ('cat', models.IntegerField(default=0)),
                ('dog', models.IntegerField(default=0)),
                ('horse', models.IntegerField(default=0)),
                ('sheep', models.IntegerField(default=0)),
                ('cow', models.IntegerField(default=0)),
                ('elephant', models.IntegerField(default=0)),
                ('bear', models.IntegerField(default=0)),
                ('zebra', models.IntegerField(default=0)),
                ('giraffe', models.IntegerField(default=0)),
                ('backpack', models.IntegerField(default=0)),
                ('umbrella', models.IntegerField(default=0)),
                ('handbag', models.IntegerField(default=0)),
                ('tie', models.IntegerField(default=0)),
                ('suitcase', models.IntegerField(default=0)),
                ('frisbee', models.IntegerField(default=0)),
                ('skis', models.IntegerField(default=0)),
                ('snowboard', models.IntegerField(default=0)),
                ('sports_ball', models.IntegerField(default=0)),
                ('kite', models.IntegerField(default=0)),
                ('baseball_bat', models.IntegerField(default=0)),
                ('baseball_glove', models.IntegerField(default=0)),
                ('skateboard', models.IntegerField(default=0)),
                ('surfboard', models.IntegerField(default=0)),
                ('tennis_racket', models.IntegerField(default=0)),
                ('bottle', models.IntegerField(default=0)),
                ('wine_glass', models.IntegerField(default=0)),
                ('cup', models.IntegerField(default=0)),
                ('fork', models.IntegerField(default=0)),
                ('knife', models.IntegerField(default=0)),
                ('spoon', models.IntegerField(default=0)),
                ('bowl', models.IntegerField(default=0)),
                ('banana', models.IntegerField(default=0)),
                ('apple', models.IntegerField(default=0)),
                ('sandwich', models.IntegerField(default=0)),
                ('orange', models.IntegerField(default=0)),
                ('broccoli', models.IntegerField(default=0)),
                ('carrot', models.IntegerField(default=0)),
                ('hot_dog', models.IntegerField(default=0)),
                ('pizza', models.IntegerField(default=0)),
                ('donut', models.IntegerField(default=0)),
                ('cake', models.IntegerField(default=0)),
                ('chair', models.IntegerField(default=0)),
                ('sofa', models.IntegerField(default=0)),
                ('pottedplant', models.IntegerField(default=0)),
                ('bed', models.IntegerField(default=0)),
                ('diningtable', models.IntegerField(default=0)),
                ('toilet', models.IntegerField(default=0)),
                ('tvmonitor', models.IntegerField(default=0)),
                ('laptop', models.IntegerField(default=0)),
                ('mouse', models.IntegerField(default=0)),
                ('remote', models.IntegerField(default=0)),
                ('keyboard', models.IntegerField(default=0)),
                ('cell_phone', models.IntegerField(default=0)),
                ('microwave', models.IntegerField(default=0)),
                ('oven', models.IntegerField(default=0)),
                ('toaster', models.IntegerField(default=0)),
                ('sink', models.IntegerField(default=0)),
                ('refrigerator', models.IntegerField(default=0)),
                ('book', models.IntegerField(default=0)),
                ('clock', models.IntegerField(default=0)),
                ('vase', models.IntegerField(default=0)),
                ('scissors', models.IntegerField(default=0)),
                ('teddy_bear', models.IntegerField(default=0)),
                ('hair_drier', models.IntegerField(default=0)),
                ('toothbrush', models.IntegerField(default=0)),
                ('potted_plant', models.IntegerField(default=0)),
                ('motorcycle', models.IntegerField(default=0)),
                ('airplane', models.IntegerField(default=0)),
                ('dining_table', models.IntegerField(default=0)),
                ('landscape', models.IntegerField(default=0)),
                ('couch', models.IntegerField(default=0)),
                ('tv', models.IntegerField(default=0)),
                ('aaa', models.IntegerField(default=0)),
                ('ford', models.IntegerField(default=0)),
                ('hao', models.IntegerField(default=0)),
                ('cj', models.IntegerField(default=0)),
                ('ywt', models.IntegerField(default=0)),
                ('unknown', models.IntegerField(default=0)),
                ('image_path', models.TextField()),
                ('image_owner', models.ForeignKey(limit_choices_to={'user_id': True}, on_delete=django.db.models.deletion.CASCADE, to='home.Members', to_field='user_id')),
            ],
            options={
                'db_table': 'classified',
                'managed': True,
            },
        ),
    ]
