import random

import pandas as pd


def table():
    with open('./static/items/name.txt', 'r', encoding='utf-8') as file:
        shampoo_names = file.read()
    shampoo_names_list = shampoo_names.split(',')
    with open('./static/items/price.txt', 'r', encoding='utf-8') as file:
        shampoo_prices = file.read()
    shampoo_prices_list = shampoo_prices.split(',')
    total = []
    for i in range(68):
        dict = {}
        dict['image'] = 'image' + str(i+1) + '.jpg'
        dict['name'] = shampoo_names_list[i]
        dict['price'] = shampoo_prices_list[i]
        total.append(dict)

    num1 = random.randint(1, 68)
    num2 = random.randint(1, 68)
    num3 = random.randint(1, 68)
    num4 = random.randint(1, 68)
    image1 = total[num1 - 1]['image']
    name1 = total[num1 - 1]['name']
    price1 = total[num1 - 1]['price']

    image2 = total[num2 - 1]['image']
    name2 = total[num2 - 1]['name']
    price2 = total[num2 - 1]['price']

    image3 = total[num3 - 1]['image']
    name3 = total[num3 - 1]['name']
    price3 = total[num3 - 1]['price']

    image4 = total[num4 - 1]['image']
    name4 = total[num4 - 1]['name']
    price4 = total[num4 - 1]['price']

    return [{'image': image1, 'name': name1, 'price': price1},
            {'image': image2, 'name': name2, 'price': price2},
            {'image': image3, 'name': name3, 'price': price3},
            {'image': image4, 'name': name4, 'price': price4}]
if __name__ == '__main__':
    result = table()
    print(result)


