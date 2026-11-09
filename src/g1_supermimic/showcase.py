"""Local browser replay with a motion selector; uses the existing physics/policy bridge."""
from types import MethodType

LABELS = {
    'shanshenbu_combo4': '四动作连续合辑（约46秒）',
    'shanshenbu': '闪身步',
    'langzi_tiqiu': '浪子踢球',
    'gouxiong_duosuomao': '狗熊躲猫猫',
    'yaotou_huangnao': '摇头晃脑',
    'shanshenbu_pro': 'Pro转体跳跃（约22秒）',
}


def play(options):
    from g1_supermimic.cli import parser
    from g1_supermimic import inference, reference
    original = reference.install_reference_ghost

    def install(env, **settings):
        ghost = original(env, **settings)
        command = env.command_manager.get_term('motion')
        lib = command.library
        available = {clip.name: i for i, clip in enumerate(lib.clips)}
        entries = [(LABELS.get(name, name), available[name]) for name in LABELS if name in available]
        entries += [(name, index) for name, index in available.items() if name not in LABELS]
        if not entries:
            raise ValueError('No replay motions found')
        state = dict(selected=entries[0][0], next=0, text=None, current=entries[0][0])
        by_label = dict(entries)

        def sample(self, n):
            if state['selected'] == '自动轮播':
                label, index = entries[state['next'] % len(entries)]
                state['next'] += 1
            else:
                label = state['selected']
                index = by_label[label]
            state['current'] = label
            if state['text'] is not None:
                state['text'].value = label
            print(f'[SHOWCASE] {label} | {lib.clips[index].name}', flush=True)
            return lib.clip_start[index].expand(n).clone()
        command._sample_frames = MethodType(sample, command)
        create_gui = command.create_gui

        def gui(self, name, server, get_env_idx, on_change=None, request_action=None):
            create_gui(name, server, get_env_idx, on_change=on_change, request_action=request_action)
            with server.gui.add_folder('南大动作选择 · 当前策略'):
                server.gui.add_markdown('蓝色为参考动作。可选择单段动作或自动轮播。')
                control = server.gui.add_dropdown('回放动作', options=[x[0] for x in entries]+['自动轮播'],
                                                  initial_value=state['selected'])
                state['text'] = server.gui.add_text('当前动作', initial_value=state['current'], disabled=True)

                @control.on_update
                def change(event):
                    state['selected'] = event.target.value
                    state['next'] = 0
                    if request_action is not None:
                        request_action('RESET')
        command.create_gui = MethodType(gui, command)
        return ghost

    reference.install_reference_ghost = install
    args = parser().parse_args(['play', '--checkpoint', str(options.checkpoint),
        '--npz-dir', str(options.npz_dir), '--allow-new-motion', '--viewer', 'viser',
        '--envs', '1', '--device', options.device, '--port', str(options.port)])
    try:
        inference.play(args)
    finally:
        reference.install_reference_ghost = original
