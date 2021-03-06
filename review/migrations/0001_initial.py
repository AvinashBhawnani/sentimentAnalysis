# Generated by Django 3.2.1 on 2021-05-06 08:05

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Product',
            fields=[
                ('asin', models.CharField(max_length=10, primary_key=True, serialize=False)),
                ('title', models.TextField()),
                ('price', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='Review',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField()),
                ('rating', models.IntegerField()),
                ('reviewer', models.TextField()),
                ('asin', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='review.product')),
            ],
        ),
    ]
