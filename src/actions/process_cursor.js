// No cursor changes outside a bounded dispatch on the recipient's UI thread.
const user32 = Process.getModuleByName('user32.dll');
const windowThread = new NativeFunction(
    user32.getExportByName('GetWindowThreadProcessId'), 'uint32', ['pointer', 'pointer']);
const dispatchMessage = new SystemFunction(
    user32.getExportByName('SendMessageTimeoutW'), 'pointer',
    ['pointer', 'uint32', 'size_t', 'ssize_t', 'uint32', 'uint32', 'pointer']);
let activeThread = 0;
let point = null;
Interceptor.attach(user32.getExportByName('GetCursorPos'), {
    onEnter(args) {
        this.output = args[0];
        this.point = this.threadId === activeThread ? point : null;
    },
    onLeave(result) {
        if (this.point !== null && !this.output.isNull()) {
            this.output.writeS32(this.point[0]);
            this.output.add(4).writeS32(this.point[1]);
            result.replace(1);
        }
    }
});
rpc.exports = {
    ready() { return true; },
    dispatch(hwnd, message, wparam, lparam, x, y) {
        if (message < 0x200 || message > 0x209)
            throw new Error('Cursor shim accepts only client mouse messages');
        const target = ptr(hwnd);
        const thread = windowThread(target, NULL);
        if (thread === 0) throw new Error('Mouse recipient no longer exists');
        point = [x, y];
        activeThread = thread;
        try {
            // SMTO_ABORTIFHUNG. Cooperative native calls let the recipient's
            // GetCursorPos hook run while this RPC waits for its window proc.
            const result = dispatchMessage(target, message, wparam, lparam,
                                           2, 2000, Memory.alloc(Process.pointerSize));
            if (result.value.isNull())
                throw new Error('SendMessageTimeoutW failed (Win32 ' + result.lastError + ')');
        } finally {
            activeThread = 0;
            point = null;
        }
    }
};
