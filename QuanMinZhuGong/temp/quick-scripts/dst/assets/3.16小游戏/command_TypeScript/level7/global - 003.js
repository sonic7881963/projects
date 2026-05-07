
                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level7/global - 003.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '47949gh4R5B/JPwE+6tmnpw', 'global - 003');
// 3.16小游戏/command_TypeScript/level7/global - 003.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global = /** @class */ (function (_super) {
    __extends(global, _super);
    function global() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    global_1 = global;
    global.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    global.prototype.onEventStart = function () {
        global_1.attack = true;
    };
    global.prototype.update = function (dt) {
        cc.log("min1, min2, min3 " + global_1.minion1_hp + " " + global_1.minion3_hp + " " + global_1.minion3_hp);
    };
    var global_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    global.attack = false;
    global.minion_attack = false;
    global.main_hp = 300;
    global.minion1_hp = 600;
    global.minion2_hp = 300;
    global.minion3_hp = 300;
    __decorate([
        property(cc.Label)
    ], global.prototype, "label", void 0);
    __decorate([
        property
    ], global.prototype, "text", void 0);
    global = global_1 = __decorate([
        ccclass
    ], global);
    return global;
}(cc.Component));
exports.default = global;

cc._RF.pop();
                    }
                    if (nodeEnv) {
                        __define(__module.exports, __require, __module);
                    }
                    else {
                        __quick_compile_project__.registerModuleFunc(__filename, function () {
                            __define(__module.exports, __require, __module);
                        });
                    }
                })();
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDdcXGdsb2JhbCAtIDAwMy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUcxQztJQUFvQywwQkFBWTtJQUFoRDtRQUFBLHFFQThCQztRQTNCRyxXQUFLLEdBQWEsSUFBSSxDQUFDO1FBR3ZCLFVBQUksR0FBVyxPQUFPLENBQUM7O0lBd0IzQixDQUFDO2VBOUJvQixNQUFNO0lBb0J2QixzQkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsWUFBWSxFQUFFLElBQUksQ0FBQyxZQUFZLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDeEQsQ0FBQztJQUNELDZCQUFZLEdBQVo7UUFDSSxRQUFNLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztJQUV6QixDQUFDO0lBQ0QsdUJBQU0sR0FBTixVQUFRLEVBQUU7UUFDTixFQUFFLENBQUMsR0FBRyxDQUFDLG1CQUFtQixHQUFDLFFBQU0sQ0FBQyxVQUFVLEdBQUMsR0FBRyxHQUFDLFFBQU0sQ0FBQyxVQUFVLEdBQUMsR0FBRyxHQUFDLFFBQU0sQ0FBQyxVQUFVLENBQUMsQ0FBQTtJQUM3RixDQUFDOztJQXJCRCx3QkFBd0I7SUFFeEIsZUFBZTtJQUVGLGFBQU0sR0FBWSxLQUFLLENBQUM7SUFDeEIsb0JBQWEsR0FBWSxLQUFLLENBQUM7SUFDL0IsY0FBTyxHQUFXLEdBQUcsQ0FBQztJQUN0QixpQkFBVSxHQUFXLEdBQUcsQ0FBQztJQUN6QixpQkFBVSxHQUFXLEdBQUcsQ0FBQztJQUN6QixpQkFBVSxHQUFXLEdBQUcsQ0FBQztJQWR0QztRQURDLFFBQVEsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDO3lDQUNJO0lBR3ZCO1FBREMsUUFBUTt3Q0FDYztJQU5OLE1BQU07UUFEMUIsT0FBTztPQUNhLE1BQU0sQ0E4QjFCO0lBQUQsYUFBQztDQTlCRCxBQThCQyxDQTlCbUMsRUFBRSxDQUFDLFNBQVMsR0E4Qi9DO2tCQTlCb0IsTUFBTSIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgZ2xvYmFsIGV4dGVuZHMgY2MuQ29tcG9uZW50IHtcclxuXHJcbiAgICBAcHJvcGVydHkoY2MuTGFiZWwpXHJcbiAgICBsYWJlbDogY2MuTGFiZWwgPSBudWxsO1xyXG5cclxuICAgIEBwcm9wZXJ0eVxyXG4gICAgdGV4dDogc3RyaW5nID0gJ2hlbGxvJztcclxuXHJcbiAgICAvLyBMSUZFLUNZQ0xFIENBTExCQUNLUzpcclxuXHJcbiAgICAvLyBvbkxvYWQgKCkge31cclxuICAgXHJcbiAgIHB1YmxpYyBzdGF0aWMgYXR0YWNrOiBib29sZWFuID0gZmFsc2U7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uX2F0dGFjazogYm9vbGVhbiA9IGZhbHNlO1xyXG4gICBwdWJsaWMgc3RhdGljIG1haW5faHA6IG51bWJlciA9IDMwMDtcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb24xX2hwOiBudW1iZXIgPSA2MDA7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uMl9ocDogbnVtYmVyID0gMzAwO1xyXG4gICBwdWJsaWMgc3RhdGljIG1pbmlvbjNfaHA6IG51bWJlciA9IDMwMDtcclxuICAgXHJcblxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICAgIHRoaXMubm9kZS5vbigndG91Y2hzdGFydCcsIHRoaXMub25FdmVudFN0YXJ0LCB0aGlzKTtcclxuICAgIH1cclxuICAgIG9uRXZlbnRTdGFydCgpIHtcclxuICAgICAgICBnbG9iYWwuYXR0YWNrID0gdHJ1ZTtcclxuICAgICAgIFxyXG4gICAgfVxyXG4gICAgdXBkYXRlIChkdCkge1xyXG4gICAgICAgIGNjLmxvZyhcIm1pbjEsIG1pbjIsIG1pbjMgXCIrZ2xvYmFsLm1pbmlvbjFfaHArXCIgXCIrZ2xvYmFsLm1pbmlvbjNfaHArXCIgXCIrZ2xvYmFsLm1pbmlvbjNfaHApXHJcbiAgICB9XHJcbn1cclxuIl19