import sqlite3

# connect, 如果没有数据文件, 就创建
con = sqlite3.connect('expdata.db')

# 创建游标
cur=con.cursor()

# 建表 sql
create_psnr= '''create table psnr(
        group_name varchar(50),
        quality int,
        psnr float
        ); '''
# cursor.execute(create_psnr)

# 查看表名
cur.execute("select name from sqlite_master where type=\'table\' order by name;")
print cur.fetchall()

# 查看表结构
cur.execute("PRAGMA table_info(psnr)")

# 读数据
import os
flist = [i for i in os.listdir('.') if os.path.splitext(i)[1].upper()=='.TXT']

import re
for each_f in flist:
    data =[]
    group_name = re.findall('_(.*)\.', each_f, re.S)[0]
    with open(each_f) as f:
        for each_l in f:
            qua, psnr = each_l.split()
            data.append((group_name, int(qua), float(psnr)))
    insert_data = "insert into psnr values (?, ?, ?)"
    # 向 psnr 表中插入数据, 批量插入, 单位为元组
    cur.executemany(insert_data, data)

con.commit()
con.close()





