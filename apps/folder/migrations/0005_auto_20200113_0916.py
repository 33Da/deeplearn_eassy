# Generated by Django 2.2.2 on 2020-01-13 01:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('folder', '0004_auto_20200113_0911'),
    ]

    operations = [
        migrations.AlterField(
            model_name='folder',
            name='last_time',
            field=models.DateTimeField(auto_now=True, help_text='最新时间', verbose_name='最新时间'),
        ),
    ]
