#[allow(unused_macros)]
#[macro_export]
macro_rules! log {
    ($($t:tt)*) => {
        #[cfg(target_arch = "wasm32")] web_sys::console::log_1(& format!($($t)*)
        .into());
    };
}
#[allow(unused_macros)]
#[macro_export]
macro_rules! convert_enum_from {
    ($src:ident, $dst:ident, $($variant:ident,)*) => {
        impl From <$src > for $dst { fn from(src : $src) -> Self { match src { $($src
        ::$variant => Self::$variant,)* } } }
    };
}
#[allow(unused_macros)]
#[macro_export]
macro_rules! convert_enum_into {
    ($src:ident, $dst:ident, $($variant:ident,)*) => {
        impl Into <$dst > for $src { fn into(self) -> $dst { match self {
        $(Self::$variant => $dst ::$variant,)* } } }
    };
}
