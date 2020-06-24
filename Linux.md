pwd: 現在path表示
ls -l
cd ..
vi → ESC → : → wq


# 管理者権限で実行
sudo su -


# Nginxを再起動
service nginx restart


# Folder削除
rm -rfd *
# File削除
rm -rfd *.*


# /home/yuenshen/nsk_edi/*　直下の全ファイルを　/usr/share/nginx/edi　にコピーする
cp -r -f /home/yuenshen/nsk_edi/* /usr/share/nginx/edi

cp -r -f /usr/share/nginx/edi.bak/vendor/ /usr/share/nginx/edi/vendor

cp -r -f /usr/share/nginx/edi.bak/.env /usr/share/nginx/edi




cp -r -f /usr/share/nginx/edi.bak/storage/fonts/* /usr/share/nginx/edi/storage/fonts/



cp -r -f -a /usr/share/nginx/edi.bak/* /usr/share/nginx/edi/



chown -R nginx:nginx edi
