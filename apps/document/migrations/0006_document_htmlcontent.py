# Generated by Django 2.2.2 on 2020-03-11 20:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('document', '0005_auto_20200127_1755'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='htmlcontent',
            field=models.TextField(blank=True, help_text='文案html内容', null=True, verbose_name='文案html内容'),
        ),
    ]