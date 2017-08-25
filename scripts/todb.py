import sqlite3


def create_table(sql, dbname='test_result.db'):
    db = sqlite3.connect(dbname, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = db.cursor()
    cur.execute(sql)
    db.commit()
    db.close()
    print('create table done...')


def insert(name, k, ssim, filesize, time, dbname='test_result.db', tbname='test'):
    '''
    # 插入值固定，不符合习惯, 改成以字典的形式插入数据
    param:
        tbname(string): name of table
        name(string): test image name, identifiy each exprience -char
        k(int): center of cluster -int
        ssim(float): quota of compress -float
        size(int): CBC file size(byte) -int
        time(float): compress consume time(ms) -float
        cbc(bin): compress file(binary) -blob
        image(np.array): reconstruct image(np.array) -array
    '''

    db = sqlite3.connect(dbname, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = db.cursor()
    sql = '''insert into %s (name, k, ssim, filesize, time) 
    values (?, ?, ?, ?, ?);''' % tbname
    cur.execute(sql, (name, k, ssim, filesize, time))
    db.commit()
    db.close()
    print('insert done...')


def select(items, dbname='test_result.db', tbname='test', name='lena'):
    '''
    params:
        items(string): select items
        name: the select condition
    return:
        res: select result
    '''
    db = sqlite3.connect(dbname, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = db.cursor()
    sql = "select %s from %s where name=\'%s\'" % (items, tbname, name)
    cur.execute(sql)
    res = cur.fetchall()
    db.close()
    return res


def main():
    sql = '''create table if not exists cbc(
    name char(20),
    k int,
    ssim float,
    filesize int,
    time float,
    primary key (name, k));'''
    name, k, ssim, size, time = 'lena', 5, 0.85, 128, 111.22
    create_table(sql)
    # insert(name, k, ssim, size, time, tbname='test3')
    # res = select('name, k, ssim, filesize', tbname='test3')[0]
    # print(res)


if __name__ == '__main__':
    main()
