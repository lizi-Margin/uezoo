from pynput import keyboard

key_state = {
    'i': False,
    'j': False,
    'k': False,
    'l': False,
    'space': False,
    '1': False,
    '2': False,
    'head_up': False,
    'head_down': False,
    'w': False,
    'a': False,
    's': False,
    'd': False,
    'q': False,
    'e': False,
}

def on_press(key):
    try:
        if key.char in key_state:
            key_state[key.char] = True
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = True
        if key == keyboard.Key.up:
            key_state['head_up'] = True
        if key == keyboard.Key.down:
            key_state['head_down'] = True


def on_release(key):
    try:
        if key.char in key_state:
            key_state[key.char] = False
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = False
        if key == keyboard.Key.up:
            key_state['head_up'] = False
        if key == keyboard.Key.down:
            key_state['head_down'] = False
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
def get_key_action_player():
    action = ([0, 0], 0, 0)
    action = list(action)  # Convert tuple to list for modification
    action[0] = list(action[0])  # Convert inner tuple to list for modification

    if key_state['i']:
        action[0][1] = 100
    if key_state['k']:
        action[0][1] = -100
    if key_state['j']:
        action[0][0] = -30
    if key_state['l']:
        action[0][0] = 30
    if key_state['space']:
        action[2] = 1
    if key_state['1']:
        action[2] = 3
    if key_state['2']:
        action[2] = 4
    if key_state['head_up']:
        action[1] = 1
    if key_state['head_down']:
        action[1] = 2

    action[0] = tuple(action[0])  # Convert inner list back to tuple
    action = tuple(action)  # Convert list back to tuple
    return action


def get_key_action_drone():
    action = [0,0,0,0]
    if key_state['w']:
        action[0] = 1
    if key_state['s']:
        action[0] = -1
    if key_state['a']:
        action[3] = -1
    if key_state['d']:
        action[3] = 1
    if key_state['e']:
        action[2]=1
    if key_state['q']:
        action[2]=-1
    return (action,)

