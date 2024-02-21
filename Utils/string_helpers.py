def snake_to_camel_case(string, capitalize_first=True):
    split_string = string.split('_')
    if capitalize_first:
        return ''.join(x.title() for x in split_string)
    return split_string[0] + ''.join(x.title() for x in split_string[1:])

