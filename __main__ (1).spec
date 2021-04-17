# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['SmartMedApp/__main__.py'],
             pathex=['/Users/projects/SmartMed'],
             binaries=[],
             datas=[('/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/dash_table', 'dash_table'),
             ('/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/dash_core_components', 'dash_core_components'),
             ('/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/dash_html_components', 'dash_html_components'),
             ('/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/plotly', 'plotly'),
             ('/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/dash_renderer', 'dash_renderer')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='SmartMed',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=False,
          icon='/Users/projects/SmartMed/iconw.icns')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='SmartMed.app',
               icon='/Users/projects/SmartMed/iconw.icns')
