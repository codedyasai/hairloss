# Generated by Django 4.1.13 on 2023-12-26 05:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bald', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='human',
            name='age',
            field=models.CharField(max_length=40),
        ),
    ]