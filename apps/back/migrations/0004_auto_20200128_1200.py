# Generated by Django 2.2.2 on 2020-01-28 12:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('back', '0003_auto_20200127_2340'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='admincountmodel',
            options={'verbose_name': '平台使用分析', 'verbose_name_plural': '平台使用分析'},
        ),
        migrations.AlterField(
            model_name='admincountmodel',
            name='data',
            field=models.DateField(auto_now_add=True, help_text='时间', verbose_name='时间'),
        ),
    ]
