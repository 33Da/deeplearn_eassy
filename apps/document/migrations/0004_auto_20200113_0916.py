# Generated by Django 2.2.2 on 2020-01-13 01:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('document', '0003_document_user'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='last_time',
            field=models.DateTimeField(auto_now=True, help_text='最新时间', verbose_name='最新时间'),
        ),
    ]
