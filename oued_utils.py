
def get_int_array_min(str_table):
    ok = False
    value = 0
    ans = []
    for c in str_table:
        if not c.isnumeric():
            if ok:
                ans.append(value)
                value = 0
            ok = False
        else:
            ok = True
            value *= 10
            value += int(c)
    assert ans
    return min(ans)
