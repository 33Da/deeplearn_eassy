# Generated by Django 2.2.2 on 2020-03-16 17:03

import DjangoUeditor.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('document', '0006_document_htmlcontent'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='content',
            field=models.TextField(blank=True, help_text='文案内容', null=True, verbose_name='文案内容'),
        ),
        migrations.AlterField(
            model_name='document',
            name='htmlcontent',
            field=DjangoUeditor.models.UEditorField(blank=True, help_text='文案html内容', null=True, verbose_name='文案html内容'),
        ),
    ]