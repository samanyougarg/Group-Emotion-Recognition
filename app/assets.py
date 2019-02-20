from flask_assets import Bundle

app_css = Bundle('app.scss', filters='scss', output='styles/app.css')

vendor_css = Bundle('vendor/material-kit.min.css', output='styles/vendor.css')

vendor_js = Bundle(
    'vendor/jquery.min.js',
    'vendor/popper.min.js',
    'vendor/bootstrap-material-design.min.js',
    'vendor/material-kit.min.js',
    'vendor/loadingoverlay.min.js',
    filters='jsmin',
    output='scripts/vendor.js')
