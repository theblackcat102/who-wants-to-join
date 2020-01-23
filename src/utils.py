def dict2table(params):
    if not isinstance(params, dict):
        params = vars(params)
    text = '\n\n'
    text = '|  Attribute  |     Value    |\n'+'|'+'-'*13+'|'+'-'*14+'|'
    for key, value in params.items():
        text += '\n|{:13}|{:14}|'.format(str(key), str(value))
    return text