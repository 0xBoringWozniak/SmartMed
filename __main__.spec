# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['SmartMedApp\\__main__.py'],
             pathex=['C:\\projects\\SmartMed'],
             binaries=[],
             datas=[('C:\\Python\\Python38\\Lib\\site-packages\\dash_table', 'dash_table'),
             ('C:\\Python\\Python38\\Lib\\site-packages\\dash_core_components', 'dash_core_components'),
             ('C:\\Python\\Python38\\Lib\\site-packages\\dash_html_components', 'dash_html_components'),
             ('C:\\Python\\Python38\\Lib\\site-packages\\plotly', 'plotly'),
             ('C:\\Python\\Python38\\Lib\\site-packages\\dash_renderer', 'dash_renderer')],
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
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='SmartMedApp',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False, icon='icon.ico')

app = BUNDLE(exe,
         name='SmartMedApp',
         icon='icon.ico',
         bundle_identifier=None,
         info_plist={'NSHighResolutionCapable': 'True'})